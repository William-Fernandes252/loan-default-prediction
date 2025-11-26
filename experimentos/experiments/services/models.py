"""Utilities for models persistence, loading and versioning."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

import joblib
from uuid_extensions import uuid7

from experiments.core.modeling.types import ModelType, Technique


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found in storage."""

    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found in storage.")


class ModelSaveError(Exception):
    """Raised when there is an error saving a model."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Failed to save model '{model_name}': {reason}")


class Classifier(Protocol):
    """Protocol for classifier models, based on scikit-learn."""

    def fit(self, X: Any, y: Any) -> None: ...

    def predict(self, X: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Represents a specific version of a model."""

    id: str
    created_at: datetime
    type: ModelType
    technique: Technique
    dataset: str


class ModelRepository(Protocol):
    """Protocol for model repository operations."""

    def save_model(self, model: Classifier, id: str | None) -> ModelVersion:
        """Saves a model and returns its version information.

        Args:
            model (Classifier): The model to save.
            id (str | None): Optional identifier for the model. If None, a new ID is generated.
        """
        ...

    def load_model(self, id: str) -> Classifier: ...

    def list_models(self) -> Iterable[ModelVersion]: ...


class ModelVersioningService(Protocol):
    """Protocol for model versioning services."""

    def get_latest_version(self, model_type: ModelType, technique: Technique) -> ModelVersion:
        """Retrieves the latest version of a specified model.

        Args:
            model_type (ModelType): The type of the model.
            technique (Technique): The technique used by the model.
        """
        ...

    def list_versions(self, model_type: ModelType, technique: Technique) -> Iterable[ModelVersion]:
        """Lists all versions of a specified model.

        Args:
            model_type (ModelType): The type of the model.
            technique (Technique): The technique used by the model.
        """
        ...

    def save_model(self, model: Classifier, id: str | None) -> ModelVersion:
        """Saves a model and returns its version information.

        Args:
            model (Classifier): The model to save.
            id (str | None): Optional identifier for the model. If None, a new ID is generated.
        """
        ...


class FileSystemModelRepository(ModelRepository):
    """A simple file system-based model repository implementation."""

    _MODEL_EXTENSION = ".joblib"
    _METADATA_EXTENSION = ".json"

    def __init__(self, base_path: Path, dataset: str, model_type: ModelType, technique: Technique):
        self.base_path = base_path
        self.dataset = dataset
        self.model_type = model_type
        self.technique = technique
        self._model_dir = self.base_path / dataset / model_type.id / technique.id
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Classifier, id: str | None) -> ModelVersion:
        model_id = id or uuid7().hex
        created_at = datetime.now(timezone.utc)
        model_path = self._model_dir / f"{model_id}{self._MODEL_EXTENSION}"
        metadata_path = self._model_dir / f"{model_id}{self._METADATA_EXTENSION}"

        try:
            joblib.dump(model, model_path)
            metadata = {
                "id": model_id,
                "created_at": created_at.isoformat(),
                "type": self.model_type.id,
                "technique": self.technique.id,
                "dataset": self.dataset,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive catch for IO errors
            if model_path.exists():
                model_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise ModelSaveError(model_id, str(exc)) from exc

        return ModelVersion(
            id=model_id,
            created_at=created_at,
            type=self.model_type,
            technique=self.technique,
            dataset=self.dataset,
        )

    def load_model(self, id: str) -> Classifier:
        model_path = self._model_dir / f"{id}{self._MODEL_EXTENSION}"
        if not model_path.exists():
            raise ModelNotFoundError(id)
        try:
            return joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - defensive catch for IO errors
            raise ModelSaveError(id, f"Could not load model: {exc}") from exc

    def list_models(self) -> Iterable[ModelVersion]:
        versions: list[ModelVersion] = []
        for metadata_file in self._model_dir.glob(f"*{self._METADATA_EXTENSION}"):
            try:
                data = json.loads(metadata_file.read_text(encoding="utf-8"))
                created_at_str = data["created_at"]
                model_id = data["id"]
                created_at = datetime.fromisoformat(created_at_str)
            except (KeyError, ValueError, json.JSONDecodeError):
                continue

            versions.append(
                ModelVersion(
                    id=model_id,
                    created_at=created_at,
                    type=self.model_type,
                    technique=self.technique,
                    dataset=data.get("dataset", self.dataset),
                )
            )

        return sorted(versions, key=lambda mv: mv.created_at, reverse=True)


class ModelVersioningServiceImpl(ModelVersioningService):
    """A simple implementation of the ModelVersioningService."""

    def __init__(self, repository: ModelRepository):
        self.repository = repository

    def get_latest_version(self, model_type: ModelType, technique: Technique) -> ModelVersion:
        versions = list(self.list_versions(model_type, technique))
        if not versions:
            raise ModelNotFoundError(f"{model_type.id}:{technique.id}")
        return versions[0]

    def list_versions(self, model_type: ModelType, technique: Technique) -> Iterable[ModelVersion]:
        versions = [
            version
            for version in self.repository.list_models()
            if version.type == model_type and version.technique == technique
        ]
        return sorted(versions, key=lambda mv: mv.created_at, reverse=True)

    def save_model(self, model: Classifier, id: str | None) -> ModelVersion:
        return self.repository.save_model(model, id)
