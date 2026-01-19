from dataclasses import dataclass
from typing import TypedDict

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import (
    Classifier,
    ClassifierFactory,
    ModelType,
    Technique,
)
from experiments.core.training.data import TrainingData, TrainingDataLoader
from experiments.core.training.splitters import DataSplitter, SplitData
from experiments.core.training.trainers import ModelTrainer, TrainedModel
from experiments.lib.pipelines import Pipeline, Task, TaskResult


@dataclass(frozen=True, slots=True)
class TrainingPipelineContext:
    """Context for training pipelines.

    Attributes:
        dataset: The dataset to train on.
        model_type: The type of model to train.
        technique: The technique for handling class imbalance.
        classifier_factory: Factory to create classifiers.
        training_data_loader: Loader for training data splits.
        trainer: The model trainer.
        data_splitter: The data splitter.
        seed: Random seed for reproducibility.
        use_gpu: Flag indicating whether to use GPU for training.
        n_jobs: Number of parallel jobs to use.
    """

    dataset: Dataset
    model_type: ModelType
    technique: Technique
    classifier_factory: ClassifierFactory
    training_data_loader: TrainingDataLoader
    trainer: ModelTrainer
    data_splitter: DataSplitter
    seed: int
    use_gpu: bool = False
    n_jobs: int = 1


class TrainingPipelineState(TypedDict, total=False):
    """State for training pipelines."""

    data_split: SplitData
    classifier: Classifier
    trained_model: TrainedModel
    training_data: TrainingData


type TrainingPipelineTask = Task[TrainingPipelineState, TrainingPipelineContext]
"""Task type for training pipelines."""


type TrainingPipelineTaskResult = TaskResult[TrainingPipelineState]
"""Task result type for training pipelines."""


type TrainingPipeline = Pipeline[TrainingPipelineState, TrainingPipelineContext]
"""Pipeline type for training pipelines."""
