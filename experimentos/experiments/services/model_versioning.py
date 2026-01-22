"""
Utilities for models persistence, loading and versioning.

This module provides a service for managing models, including saving, loading, and listing model versions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Protocol

from pydantic import UUID7, BaseModel, Field

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


class ModelVersion(BaseModel):
    """Represents a specific version of a model."""

    id: Annotated[
        UUID7,
        Field(
            json_schema_extra={"description": "Unique identifier for the model version."},
        ),
    ]
    created_at: Annotated[
        datetime,
        Field(
            json_schema_extra={
                "description": "The timestamp when the model version was created.",
            }
        ),
    ]
    type: Annotated[
        ModelType,
        Field(json_schema_extra={"description": "The type of model."}),
    ]
    technique: Annotated[
        Technique,
        Field(json_schema_extra={"description": "The technique used to handle class imbalance."}),
    ]
    dataset: Annotated[
        Dataset,
        Field(json_schema_extra={"description": "The dataset associated with the model version."}),
    ]


class ModelVersionsQuery(BaseModel):
    """Query parameters for listing model versions."""

    model_type: Annotated[
        ModelType | None,
        Field(json_schema_extra={"description": "The type of model used."}),
    ] = None
    technique: Annotated[
        Technique | None,
        Field(
            json_schema_extra={"description": "The technique used to handle class imbalance."},
        ),
    ] = None


class ModelVersionListResult(BaseModel):
    """Result of listing model versions."""

    versions: Annotated[
        list[ModelVersion],
        Field(json_schema_extra={"description": "List of model versions."}, default_factory=list),
    ]
    total_count: Annotated[
        int,
        Field(json_schema_extra={"description": "Total number of model versions available."}),
    ]


@dataclass(frozen=True, slots=True)
class ModelVersionRetrieveResult:
    """Result of retrieving a specific model version."""

    version: ModelVersion
    """Version information of the model."""

    model: TrainedModel
    """The trained model associated with the version."""


class ModelRepository(Protocol):
    """Protocol for model repository operations."""

    def save_model(
        self,
        model: Classifier,
        id: str | None = None,
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

    def list_models(
        self, dataset: Dataset, params: ModelVersionsQuery | None = None
    ) -> ModelVersionListResult:
        """Lists all models in the repository.

        Args:
            dataset (Dataset): The dataset associated with the models.
            params (ModelVersionsQuery | None): Optional query parameters to filter the models

        Returns:
            ModelVersionListResult: The list of model versions and total count.
        """
        ...

    def get_version(self, dataset: Dataset, id: str) -> ModelVersionRetrieveResult:
        """Retrieves a specific model version for a dataset by its identifier.

        Args:
            dataset (Dataset): The dataset associated with the model version.
            id (str): The identifier of the model version.

        Returns:
            ModelVersion: The model version information.

        Raises:
            ModelNotFoundError: If no model is found for the given identifier and dataset.
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
        self, dataset: Dataset, query: ModelVersionsQuery
    ) -> ModelVersionRetrieveResult:
        """Retrieves the latest version of a specified model.

        Args:
            dataset (Dataset): The dataset associated with the model.
            query (ModelVersionsQuery): The query parameters specifying model type and technique.

        Returns:
            ModelVersionRetrieveResult: The latest model version and its trained model.

        Raises:
            ModelNotFoundError: If no model is found for the given identifier and dataset.
        """
        versions = self.repository.list_models(dataset, query).versions
        if not versions:
            raise ModelNotFoundError(
                f"No versions found for model type '{query.model_type}' "
                f"and technique '{query.technique}'"
            )

        latest_version = max(versions, key=lambda mv: mv.created_at)
        return self.repository.get_version(dataset, str(latest_version.id))

    def get_version(self, dataset: Dataset, id: str) -> ModelVersionRetrieveResult:
        """Retrieves a specific model version by its identifier.

        Args:
            id (str): The identifier of the model version.

        Returns:
            ModelVersion: The model version information.

        Raises:
            ModelNotFoundError: If no model is found for the given identifier and dataset.
        """
        return self.repository.get_version(dataset, id)

    def list_versions(
        self, dataset: Dataset, query: ModelVersionsQuery | None = None
    ) -> ModelVersionListResult:
        """Lists all model versions for a given dataset.

        Args:
            dataset (Dataset): The dataset associated with the models.
            query (ModelVersionsQuery | None): Optional query parameters to filter the models.

        Returns:
            ModelVersionListResult: The list of model versions and total count.
        """
        return self.repository.list_models(dataset, query)

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
            dataset=params.dataset,
            model_type=params.model_type,
            technique=params.technique,
            params=trained_model.params,
            seed=trained_model.seed,
        )

        return trained_model, model_version


class TrainedModelLoaderImpl:
    """Implementation of `TrainedModelLoader` using the model versioning service."""

    def __init__(self, model_versioner: ModelVersioner) -> None:
        self._model_versioner = model_versioner

    def load_model(self, dataset: Dataset, model_id: str | None) -> TrainedModel:
        """Retrieve a trained model by its identifier.

        Args:
            dataset (Dataset): The dataset associated with the trained model.
            model_id (str | None): The identifier of the trained model, or None to load the latest model.

        Returns:
            The loaded trained model.

        Raises:
            ValueError: If no model is found for the given identifier and dataset.
        """
        try:
            if model_id is None:
                version_result = self._model_versioner.get_latest_version(
                    dataset,
                    ModelVersionsQuery(model_type=None, technique=None),
                )
            else:
                version_result = self._model_versioner.get_version(dataset, model_id)
        except ModelNotFoundError as e:
            raise ValueError("No trained model found for the given identifier and dataset.") from e
        else:
            return version_result.model
