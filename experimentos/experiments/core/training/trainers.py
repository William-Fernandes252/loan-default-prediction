from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from experiments.core.modeling.classifiers import Classifier, ModelType, Technique
from experiments.core.training.splitters import SplitData


@dataclass(slots=True, frozen=True)
class ModelTrainRequest:
    """Request data for model training.

    Attributes:
        classifier (Classifier): The classifier to be trained.
        model_type (ModelType): The type of the model.
        technique (Technique): The technique used by the model.
        data (SplitData): The split training/test data.
        seed (int): Random seed for reproducibility.
    """

    classifier: Classifier
    model_type: ModelType
    technique: Technique
    data: SplitData
    seed: int


@dataclass(slots=True, frozen=True)
class TrainedModel:
    """Result of model training.

    Attributes:
        model (Classifier): The trained classifier.
        params: The hyperparameters used for training.
        seed (int): The random seed used during training.
    """

    model: Classifier
    params: dict[str, Any]
    seed: int


@runtime_checkable
class ModelTrainer(Protocol):
    """Protocol for training and optimizing models."""

    def train(
        self,
        request: ModelTrainRequest,
        n_jobs: int = 1,
    ) -> TrainedModel:
        """Train and optimize a model.

        Args:
            request (ModelTrainRequest): The training request data.
            n_jobs (int): Number of parallel jobs for training. Defaults to `1`.

        Returns:
            The trained model with best parameters.
        """
        ...
