"""Defines the inference service."""

from typing import Annotated

from pydantic import BaseModel

from experiments.core.data.datasets import Dataset
from experiments.core.training.data import TrainingDataLoader
from experiments.core.training.splitters import DataSplitter
from experiments.core.training.trainers import TrainedModelLoader
from experiments.lib.pipelines import PipelineExecutor
from experiments.pipelines.predictions.factory import PredictionsPipelineFactory
from experiments.pipelines.predictions.pipeline import (
    PredictionsPipelineContext,
    PredictionsPipelineState,
)
from experiments.services.training_executor import SeedGenerator


class InferenceResult(BaseModel):
    """Result of an inference operation."""

    predictions: Annotated[list[float], "Predicted values from the model"]


class InferenceService:
    """Service to execute inference pipelines."""

    def __init__(
        self,
        pipeline_executor: PipelineExecutor,
        predictions_pipeline_factory: PredictionsPipelineFactory,
        training_data_loader: TrainingDataLoader,
        trained_model_loader: TrainedModelLoader,
        data_splitter: DataSplitter,
        seed_generator: SeedGenerator,
    ) -> None:
        self._pipeline_executor = pipeline_executor
        self._predictions_pipeline_factory = predictions_pipeline_factory
        self._training_data_loader = training_data_loader
        self._trained_model_loader = trained_model_loader
        self._data_splitter = data_splitter
        self._seed_generator = seed_generator

    def run_inference_on_test_set(
        self, dataset: Dataset, model_id: str | None = None
    ) -> InferenceResult:
        """Run inference pipeline on the test set of the given dataset.

        Args:
            dataset: The dataset to run inference on.
            model_id: Optional identifier for the model to use. If None, the latest
                      trained model for the dataset will be used.

        Returns:
            InferenceResult containing the predictions.
        """
        seed = self._seed_generator()

        context = PredictionsPipelineContext(
            model_id=model_id,
            dataset=dataset,
            training_data_loader=self._training_data_loader,
            trained_model_loader=self._trained_model_loader,
            data_splitter=self._data_splitter,
            seed=seed,
        )

        pipeline = self._predictions_pipeline_factory.create_pipeline()
        result = self._pipeline_executor.execute(pipeline, PredictionsPipelineState(), context)

        if not result.succeeded():
            raise RuntimeError(f"Inference pipeline failed: {result.last_error()}")

        return InferenceResult(predictions=result.final_state["predictions"].tolist())
