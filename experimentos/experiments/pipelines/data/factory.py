from enum import Enum
from typing import cast

from experiments.core.data import (
    Dataset,
    get_transformer_registry,
)
from experiments.core.data.registry import TransformerRegistry
from experiments.core.data.repository import DataRepository
from experiments.core.data.transformer import Transformer
from experiments.core.modeling.features import extract_features_and_target
from experiments.lib.pipelines import Pipeline, Task, TaskResult, TaskStatus
from experiments.pipelines.data.context import DataPipelineContext
from experiments.pipelines.data.exporters import (
    export_final_features_as_parquet,
    export_processed_data_as_parquet,
)
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
    EXTRACT_FINAL_FEATURES = "ExtractFinalFeatures"
    EXPORT_FINAL_FEATURES = "ExportFinalFeatures"


def check_already_processed(
    state: DataPipelineState, context: DataPipelineContext
) -> TaskResult[DataPipelineState]:
    dataset = context.dataset
    if context.data_repository.is_processed(dataset):
        return TaskResult(
            state,
            TaskStatus.SKIPPED,
            "Processed data already exists.",
        )
    return TaskResult(state, TaskStatus.SUCCESS, "No existing processed data found.")


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
            A configured `DataProcessingPipeline` instance.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """

        context = DataPipelineContext(
            dataset=dataset,
            data_repository=self._data_repository,
            use_gpu=use_gpu,
        )

        pipeline = Pipeline[DataPipelineState, DataPipelineContext](
            f"DataProcessingPipeline[dataset={dataset}]", context
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
        pipeline.add_step(
            DataProcessingPipelineSteps.EXTRACT_FINAL_FEATURES.value,
            self._create_feature_extractor(),
        )
        pipeline.add_step(
            DataProcessingPipelineSteps.EXPORT_FINAL_FEATURES.value,
            export_final_features_as_parquet,
        )

        return pipeline

    def _create_transformer(
        self,
        dataset: Dataset,
    ) -> Task[DataPipelineState, DataPipelineContext]:
        """Create a transformer step for the data processing pipeline.

        Args:
            dataset: The dataset to create the transformer for.
            transformer_registry: Registry mapping dataset IDs to transformer classes.

        Returns:
            A Step that applies the appropriate transformer for the dataset.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """
        transformer = self._transformer_registry.get(dataset.value)
        if transformer is None:
            raise ValueError(f"No transformer registered for dataset: '{dataset.value}'")

        def transform(
            state: DataPipelineState, context: DataPipelineContext
        ) -> TaskResult[DataPipelineState]:
            if state["raw_data"] is None:
                return TaskResult(
                    state, TaskStatus.FAILURE, "No raw data found in state for transformation."
                )
            state["interim_data"] = cast(Transformer, transformer)(
                state["raw_data"], context.use_gpu
            )
            return TaskResult(state, TaskStatus.SUCCESS, "Data transformation successful.")

        return transform

    def _create_feature_extractor(
        self,
    ) -> Task[DataPipelineState, DataPipelineContext]:
        """Create a feature extractor step for the data processing pipeline."""

        def extract_features(
            state: DataPipelineState, context: DataPipelineContext
        ) -> TaskResult[DataPipelineState]:
            if state["interim_data"] is None:
                return TaskResult(
                    state,
                    TaskStatus.FAILURE,
                    "No interim data found in state for feature extraction.",
                )

            state["X_final"], state["y_final"] = extract_features_and_target(state["interim_data"])
            return TaskResult(state, TaskStatus.SUCCESS, "Feature extraction successful.")

        return extract_features
