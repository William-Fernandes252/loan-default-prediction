from enum import Enum

from experiments.core.data import (
    TransformerRegistry,
    get_transformer_registry,
)
from experiments.core.modeling.features import FeatureExtractor
from experiments.lib.pipelines import Pipeline, Task, TaskResult, TaskStatus
from experiments.pipelines.data.base import (
    DataProcessingPipeline,
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)
from experiments.pipelines.data.exporters import (
    export_final_features_as_parquet,
    export_processed_data_as_parquet,
)
from experiments.pipelines.data.loaders import load_raw_data_from_csv


class DataProcessingPipelineSteps(Enum):
    """Enumeration of data processing pipeline steps."""

    CHECK_ALREADY_PROCESSED = "CheckAlreadyProcessed"
    LOAD_RAW_DATA = "LoadRawData"
    TRANSFORM_DATA = "TransformData"
    EXPORT_PROCESSED_DATA = "ExportProcessedData"
    EXTRACT_FINAL_FEATURES = "ExtractFinalFeatures"
    EXPORT_FINAL_FEATURES = "ExportFinalFeatures"


def check_already_processed(
    state: DataProcessingPipelineState, context: DataProcessingPipelineContext
) -> TaskResult[DataProcessingPipelineState]:
    dataset = context.dataset
    state["is_processed"] = context.data_repository.is_processed(dataset)
    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        "Data already processed." if state["is_processed"] else "Data not yet processed.",
    )


def run_if_not_processed(
    state: DataProcessingPipelineState, context: DataProcessingPipelineContext
):
    if state.get("is_processed", True) and not context.force_overwrite:
        return False, "Data already processed; skipping step."
    return True, None


class DataProcessingPipelineFactory:
    """Factory for creating configured data pipelines.

    This factory creates pipelines with the appropriate transformer
    for each dataset, handling the wiring of loaders and exporters
    using the storage layer.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        transformer_registry: TransformerRegistry | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            transformer_registry: Optional registry of transformers. If not provided, the global registry is used.
        """
        self._transformer_registry = (
            transformer_registry
            if transformer_registry is not None
            else get_transformer_registry()
        )
        self._feature_extractor = feature_extractor

    def create(
        self,
        name: str = "DataProcessingPipeline",
    ) -> DataProcessingPipeline:
        """Create a pipeline configured for the specified dataset.

        Args:
            name: The name of the pipeline.

        Returns:
            A configured `DataProcessingPipeline` instance.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """

        pipeline = Pipeline[DataProcessingPipelineState, DataProcessingPipelineContext](
            name=name,
        )

        pipeline.add_step(
            DataProcessingPipelineSteps.CHECK_ALREADY_PROCESSED.value,
            check_already_processed,
        )
        pipeline.add_conditional_step(
            DataProcessingPipelineSteps.LOAD_RAW_DATA.value,
            load_raw_data_from_csv,
            run_if_not_processed,
        )
        pipeline.add_conditional_step(
            DataProcessingPipelineSteps.TRANSFORM_DATA.value,
            self._create_transformer(),
            run_if_not_processed,
        )
        pipeline.add_conditional_step(
            DataProcessingPipelineSteps.EXPORT_PROCESSED_DATA.value,
            export_processed_data_as_parquet,
            run_if_not_processed,
        )
        pipeline.add_conditional_step(
            DataProcessingPipelineSteps.EXTRACT_FINAL_FEATURES.value,
            self._create_feature_extractor(),
            run_if_not_processed,
        )
        pipeline.add_conditional_step(
            DataProcessingPipelineSteps.EXPORT_FINAL_FEATURES.value,
            export_final_features_as_parquet,
            run_if_not_processed,
        )

        return pipeline

    def _create_transformer(
        self,
    ) -> Task[DataProcessingPipelineState, DataProcessingPipelineContext]:
        """Create a transformer step for the data processing pipeline.

        Returns:
            A Step that applies the appropriate transformer for the dataset.

        Raises:
            ValueError: If no transformer is registered for the dataset.
        """

        def transform(
            state: DataProcessingPipelineState, context: DataProcessingPipelineContext
        ) -> TaskResult[DataProcessingPipelineState]:
            transformer = self._transformer_registry.get(context.dataset)
            if transformer is None:
                raise ValueError(f"No transformer registered for dataset: '{context.dataset}'")

            if state["raw_data"] is None:
                return TaskResult(
                    state, TaskStatus.FAILURE, "No raw data found in state for transformation."
                )
            state["interim_data"] = transformer(state["raw_data"], context.use_gpu)
            return TaskResult(state, TaskStatus.SUCCESS, "Data transformation successful.")

        return transform

    def _create_feature_extractor(
        self,
    ) -> Task[DataProcessingPipelineState, DataProcessingPipelineContext]:
        """Create a feature extractor step for the data processing pipeline."""

        def extract_features(
            state: DataProcessingPipelineState, context: DataProcessingPipelineContext
        ) -> TaskResult[DataProcessingPipelineState]:
            if state["interim_data"] is None:
                return TaskResult(
                    state,
                    TaskStatus.FAILURE,
                    "No interim data found in state for feature extraction.",
                )

            state["X_final"], state["y_final"] = (
                self._feature_extractor.extract_features_and_target(state["interim_data"])
            )
            return TaskResult(state, TaskStatus.SUCCESS, "Feature extraction successful.")

        return extract_features
