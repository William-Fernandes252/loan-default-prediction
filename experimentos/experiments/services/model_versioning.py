"""
Utilities for models persistence, loading and versioning.

This module provides a service for managing models, including saving, loading, and listing model versions.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple, Protocol

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import Classifier, ModelType, Technique
from experiments.core.training.trainers import TrainedModel
from experiments.services.training_executor import TrainingExecutor, TrainModelParams


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found in storage."""

    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found in storage.")


class ModelSaveError(Exception):
    """Raised when there is an error saving a model."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Failed to save model '{model_name}': {reason}")


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Represents a specific version of a model."""

    id: str
    created_at: datetime
    type: ModelType
    technique: Technique
    dataset: Dataset


class ModelVersionListResult(NamedTuple):
    """Result of listing model versions."""

    versions: Iterable[ModelVersion]
    total_count: int


class ModelRepository(Protocol):
    """Protocol for model repository operations."""

    def save_model(
        self,
        model: Classifier,
        id: str | None,
        *,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        params: dict,
        seed: int,
    ) -> ModelVersion:
        """Saves a model and returns its version information.

        Args:
            model (Classifier): The model to save.
            id (str | None): Optional identifier for the model. If None, a new ID is generated.
        """
        ...

    def load_model(self, id: str) -> TrainedModel:
        """Loads a model by its identifier.

        Args:
            id (str): The identifier of the model to load.

        Returns:
            TrainedModel: The loaded model.
        """
        ...

    def list_models(self) -> ModelVersionListResult:
        """Lists all models in the repository.

        Returns:
            ModelVersionListResult: The list of model versions and total count.
        """
        ...


class ModelVersioner:
    """Service for managing model versions."""

    def __init__(
        self,
        model_repository: ModelRepository,
        training_executor: TrainingExecutor,
    ) -> None:
        self.repository = model_repository
        self._training_executor = training_executor

    def get_latest_version(
        self, model_type: ModelType, technique: Technique
    ) -> tuple[TrainedModel, ModelVersion]:
        """Retrieves the latest version of a specified model.

        Args:
            model_type (ModelType): The type of the model.
            technique (Technique): The technique used by the model.
        """
        versions = list(self.list_versions(model_type, technique))
        if not versions:
            raise ModelNotFoundError(f"{model_type}:{technique}")
        latest_version = versions[0]
        latest_model = self.repository.load_model(latest_version.id)
        return latest_model, latest_version

    def list_versions(self, model_type: ModelType, technique: Technique) -> Iterable[ModelVersion]:
        """Lists all versions of a specified model.

        Args:
            model_type (ModelType): The type of the model.
            technique (Technique): The technique used by the model.
        """
        versions = [
            version
            for version in self.repository.list_models().versions
            if version.type == model_type and version.technique == technique
        ]
        return sorted(versions, key=lambda mv: mv.created_at, reverse=True)

    def train_new_version(
        self,
        params: TrainModelParams,
    ) -> tuple[TrainedModel, ModelVersion]:
        """Trains a new model version and saves it to the repository.

        Args:
            params (TrainModelParams): Parameters for training the model.

        Returns:
            tuple[TrainedModel, ModelVersion]: The trained model and its version information.
        """
        trained_model = self._training_executor.train_model(params)

        model_version = self.repository.save_model(
            model=trained_model.model,
            id=None,
            dataset=params.dataset,
            model_type=params.model_type,
            technique=params.technique,
            params=trained_model.params,
            seed=trained_model.seed,
        )

        return trained_model, model_version
