"""Defines an implementation for the model repository interface using a storage backend."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import uuid

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import Classifier, ModelType, Technique
from experiments.core.training.trainers import TrainedModel
from experiments.services.model_versioning import (
    ModelNotFoundError,
    ModelVersion,
    ModelVersionListResult,
)
from experiments.storage import Storage
from experiments.storage.errors import FileDoesNotExistError


@dataclass(frozen=True, slots=True)
class ModelStorageLayout:
    """Model layout on the storage for model repository operations."""

    model_key_template: str = "models/{dataset}/{model_type}/{technique}/{model_id}.joblib"
    model_prefix: str = "models/"

    def get_model_key(
        self,
        dataset: str,
        model_type: str,
        technique: str,
        model_id: str,
    ) -> str:
        """Get the model key for a given dataset ID, model type, technique, and model ID."""
        return self.model_key_template.format(
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            model_id=model_id,
        )

    def parse_model_key(self, key: str) -> dict[str, str] | None:
        """Parse a model key and extract its components.

        Args:
            key: The storage key to parse.

        Returns:
            A dictionary with dataset, model_type, technique, and model_id,
            or None if the key doesn't match the expected pattern.
        """
        pattern = r"^models/([^/]+)/([^/]+)/([^/]+)/([^/]+)\.joblib$"
        match = re.match(pattern, key)
        if not match:
            return None
        return {
            "dataset": match.group(1),
            "model_type": match.group(2),
            "technique": match.group(3),
            "model_id": match.group(4),
        }


@dataclass(frozen=True, slots=True)
class StoredModel:
    """Represents a model stored in the repository."""

    model: Classifier
    params: dict
    seed: int
    dataset: Dataset
    model_type: ModelType
    technique: Technique


class ModelStorageRepository:
    """Model repository implementation using a storage backend."""

    def __init__(
        self,
        storage: Storage,
        layout: ModelStorageLayout | None = None,
    ):
        self._storage = storage
        self._layout = layout or ModelStorageLayout()

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
        """Saves a model to the storage backend.

        Args:
            model: The model to save.
            id: Optional identifier for the model. If None, a new ID is generated.
            dataset: The dataset the model was trained on.
            model_type: The type of the model.
            technique: The technique used for training.
            params: Optional hyperparameters used for training.
            seed: The random seed used during training.

        Returns:
            ModelVersion: The version information of the saved model.
        """
        model_id = id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        key = self._layout.get_model_key(
            dataset=dataset.value,
            model_type=model_type.value,
            technique=technique.value,
            model_id=model_id,
        )

        stored_model = StoredModel(
            model=model,
            params=params or {},
            seed=seed,
            dataset=dataset,
            model_type=model_type,
            technique=technique,
        )

        self._storage.write_joblib(stored_model, key)

        return ModelVersion(
            id=model_id,
            created_at=created_at,
            type=model_type,
            technique=technique,
            dataset=dataset,
        )

    def load_model(self, id: str) -> TrainedModel:
        """Loads a model from the storage backend.

        Args:
            id: The identifier of the model to load.

        Returns:
            TrainedModel: The loaded model.

        Raises:
            ModelNotFoundError: If no model with the given ID is found.
        """
        # Search for the model across all datasets, model types, and techniques
        for file_info in self._storage.list_files(self._layout.model_prefix, "*.joblib"):
            parsed = self._layout.parse_model_key(file_info.key)
            if parsed and parsed["model_id"] == id:
                try:
                    stored_model: StoredModel = self._storage.read_joblib(file_info.key)
                    return TrainedModel(
                        model=stored_model.model,
                        params=stored_model.params,
                        seed=stored_model.seed,
                    )
                except FileDoesNotExistError:
                    raise ModelNotFoundError(id)

        raise ModelNotFoundError(id)

    def list_models(self) -> ModelVersionListResult:
        """Lists all models in the repository.

        Returns:
            ModelVersionListResult: The list of model versions and total count.
        """
        versions: list[ModelVersion] = []

        for file_info in self._storage.list_files(self._layout.model_prefix, "*.joblib"):
            parsed = self._layout.parse_model_key(file_info.key)
            if parsed:
                try:
                    dataset = Dataset(parsed["dataset"])
                    model_type = ModelType(parsed["model_type"])
                    technique = Technique(parsed["technique"])

                    version = ModelVersion(
                        id=parsed["model_id"],
                        created_at=file_info.last_modified or datetime.now(timezone.utc),
                        type=model_type,
                        technique=technique,
                        dataset=dataset,
                    )
                    versions.append(version)
                except ValueError:
                    # Skip files that don't match valid enum values
                    continue

        return ModelVersionListResult(versions=versions, total_count=len(versions))
