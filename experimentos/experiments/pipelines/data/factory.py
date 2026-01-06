from enum import Enum
from typing import cast

from experiments.core.data_new import (
    Dataset,
    get_transformer_registry,
)
from experiments.core.data_new.registry import TransformerRegistry
from experiments.core.data_new.repository import DataRepository
from experiments.core.data_new.transformer import Transformer
from experiments.lib.pipelines import Pipeline, errors
from experiments.lib.pipelines.steps import Runnable
from experiments.pipelines.data.context import DataPipelineContext
from experiments.pipelines.data.exporters import export_processed_data_as_parquet
from experiments.pipelines.data.loaders import load_raw_data_from_csv
from experiments.pipelines.data.state import DataPipelineState

type DataProcessingPipeline = Pipeline[DataPipelineState, DataPipelineContext]
"""Data processing pipeline type alias."""


class DataProcessingPipelineSteps(Enum):
    """Enumeration of data processing pipeline steps."""

    CHECK_ALREADY_PROCESSED = "CheckAlreadyProcessed"
    LOAD_RAW_DATA = "LoadRawData"
    TRANSFORM_DATA = "TransformData"
    EXPORT_PROCESSED_DATA = "ExportProcessedData"


def check_already_processed(
    state: DataPipelineState, context: DataPipelineContext
) -> DataPipelineState:
    dataset = context.dataset
    if context.data_repository.is_processed(dataset):
        raise errors.PipelineInterruption(
            f"Processed data for dataset {dataset.display_name} already exists. Overwrite?",
            context,
        )
    return state


class DataProcessingPipelineFactory:
    """Factory for creating configured data pipelines.

    This factory creates pipelines with the appropriate transformer
    for each dataset, handling the wiring of loaders and exporters
    using the storage layer.
    """

    def __init__(
        self,
        data_repository: DataRepository,
        transformer_registry: TransformerRegistry | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            data_repository: The data repository for data operations.
            transformer_registry: Optional registry of transformers. If not provided, the global registry is used.
        """
        self._data_repository = data_repository

        # Use provided registry or get from global registration system
        self._transformer_registry = (
            transformer_registry
            if transformer_registry is not None
            else get_transformer_registry()
        )

    def create(
        self,
        dataset: Dataset,
        use_gpu: bool,
        force_overwrite: bool,
    ) -> DataProcessingPipeline:
        """Create a pipeline configured for the specified dataset.

        Args:
            dataset: The dataset to create a pipeline for.
            use_gpu: Whether to enable GPU acceleration for transformations.
            force_overwrite: Whether to overwrite existing processed data.
            error_handlers: Optional mapping of pipeline steps to error handler functions.

        Returns:
            A configured DataProcessingPipeline instance.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """

        context = DataPipelineContext(
            dataset=dataset,
            data_repository=self._data_repository,
            use_gpu=use_gpu,
        )

        pipeline = Pipeline[DataPipelineState, DataPipelineContext](
            f"DataProcessingPipeline_{dataset.id}", context
        )

        if not force_overwrite:
            pipeline.add_step(
                DataProcessingPipelineSteps.CHECK_ALREADY_PROCESSED.value, check_already_processed
            )

        pipeline.add_step(DataProcessingPipelineSteps.LOAD_RAW_DATA.value, load_raw_data_from_csv)
        pipeline.add_step(
            DataProcessingPipelineSteps.TRANSFORM_DATA.value,
            self._create_transformer(dataset),
        )
        pipeline.add_step(
            DataProcessingPipelineSteps.EXPORT_PROCESSED_DATA.value,
            export_processed_data_as_parquet,
        )

        return pipeline

    def _create_transformer(
        self,
        dataset: Dataset,
    ) -> Runnable[DataPipelineState, DataPipelineContext]:
        """Create a transformer step for the data processing pipeline.

        Args:
            dataset: The dataset to create the transformer for.
            transformer_registry: Registry mapping dataset IDs to transformer classes.

        Returns:
            A Step that applies the appropriate transformer for the dataset.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """
        transformer = self._transformer_registry.get(dataset.id)
        if transformer is None:
            raise ValueError(f"No transformer registered for dataset: {dataset.id}")

        def transform(state: DataPipelineState, context: DataPipelineContext) -> DataPipelineState:
            raw_data = state["raw_data"]
            try:
                processed_data = cast(Transformer, transformer)(raw_data, context.use_gpu)
            except Exception as e:
                raise errors.PipelineException(
                    f"Data transformation failed for dataset {dataset.id}: {e}"
                ) from e
            state["interim_data"] = processed_data
            return state

        return transform
