"""Model versioning service factory.

This module provides a factory for creating ModelVersioningService instances,
replacing the method that was previously on the Context class.
"""

from pathlib import Path

from experiments.core.modeling.types import ModelType, Technique
from experiments.services.models import (
    FileSystemModelRepository,
    ModelVersioningService,
    ModelVersioningServiceImpl,
)


class ModelVersioningServiceFactory:
    """Factory for creating ModelVersioningService instances.

    This factory creates versioning services configured for specific
    dataset/model/technique combinations, implementing the
    ModelVersioningProvider protocol.
    """

    def __init__(self, models_dir: Path) -> None:
        """Initialize the factory.

        Args:
            models_dir: Root directory for model storage.
        """
        self._models_dir = models_dir

    def get_model_versioning_service(
        self,
        dataset_id: str,
        model_type: ModelType,
        technique: Technique,
    ) -> ModelVersioningService:
        """Create a ModelVersioningService for the given configuration.

        Args:
            dataset_id: The dataset identifier.
            model_type: The type of model.
            technique: The technique for handling class imbalance.

        Returns:
            A configured ModelVersioningService instance.
        """
        repo = FileSystemModelRepository(
            self._models_dir,
            dataset_id,
            model_type,
            technique,
        )
        return ModelVersioningServiceImpl(repo)
