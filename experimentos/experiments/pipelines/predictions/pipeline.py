from dataclasses import dataclass
from typing import TypedDict

import numpy as nd

from experiments.core.data.datasets import Dataset
from experiments.core.training.data import TrainingData, TrainingDataLoader
from experiments.core.training.splitters import DataSplitter, SplitData
from experiments.core.training.trainers import TrainedModel, TrainedModelLoader
from experiments.lib.pipelines import Pipeline, Task, TaskResult


@dataclass(frozen=True, slots=True)
class PredictionsPipelineContext:
    """Context for prediction pipelines.

    Attributes:
        model_id (str | None): Optional identifier for the model to be used for predictions. If not provided, the latest model trained with the given dataset will be used.
        dataset (Dataset): The dataset to be used for predictions.
        training_data_loader (TrainingDataLoader): Loader for training data.
        trained_model_loader (TrainedModelLoader): Loader for trained models.
        data_splitter (DataSplitter): Splitter for training and test data.
        seed (int): Random seed for reproducibility.
    """

    model_id: str | None
    dataset: Dataset
    training_data_loader: TrainingDataLoader
    trained_model_loader: TrainedModelLoader
    data_splitter: DataSplitter
    seed: int


class PredictionsPipelineState(TypedDict, total=False):
    """State for prediction pipelines."""

    data_split: SplitData
    trained_model: TrainedModel
    training_data: TrainingData
    predictions: nd.ndarray


type PredictionsPipelineTask = Task[PredictionsPipelineState, PredictionsPipelineContext]
"""Task type for prediction pipelines."""

type PredictionsPipelineTaskResult = TaskResult[PredictionsPipelineState]
"""Task result type for prediction pipelines."""


type PredictionsPipeline = Pipeline[PredictionsPipelineState, PredictionsPipelineContext]
"""Pipeline type for prediction pipelines."""
