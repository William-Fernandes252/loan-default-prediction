"""Experiment persistence implementations for the experiment pipeline."""

from typing import Any

from loguru import logger
import polars as pl
from sklearn.base import BaseEstimator

from experiments.core.experiment.protocols import ExperimentContext
from experiments.services.model_versioning import (
    ModelVersioningServiceFactory,
)
from experiments.services.storage import StorageService


class ParquetExperimentPersister:
    """Persists experiment results to parquet files and models via versioning service.

    This implementation:
    - Saves metrics to parquet checkpoint files using storage layer
    - Uses ModelVersioningService for model persistence
    - Manages checkpoint lifecycle (exists, discard)
    """

    def __init__(
        self,
        storage: StorageService,
        model_versioning_service_factory: ModelVersioningServiceFactory | None = None,
    ) -> None:
        """Initialize the persister.

        Args:
            storage: Storage service for file operations.
            model_versioning_service_factory: Optional factory for model versioning services.
        """
        self._storage = storage
        self._model_versioning_service_factory = model_versioning_service_factory

    def save_checkpoint(
        self,
        metrics: dict[str, Any],
        checkpoint_uri: str,
    ) -> None:
        """Save experiment results to a checkpoint file.

        Args:
            metrics: Metrics dictionary to save.
            checkpoint_uri: URI to save the checkpoint.
        """
        df = pl.DataFrame([metrics])
        self._storage.write_parquet(df, checkpoint_uri)

    def save_model(
        self,
        model: BaseEstimator,
        context: ExperimentContext,
    ) -> None:
        """Save a trained model using the versioning service.

        Args:
            model: The trained model to save.
            context: Experiment context containing dataset, model type, and technique.
        """
        if self._model_versioning_service_factory is None:
            return

        try:
            # Create specific versioning service for this experiment
            versioning_service = (
                self._model_versioning_service_factory.get_model_versioning_service(
                    dataset_id=context.identity.dataset.id,
                    model_type=context.identity.model_type,
                    technique=context.identity.technique,
                )
            )
            versioning_service.save_model(model, None)
        except Exception as e:
            logger.warning(f"Model save failed: {e}")

    def checkpoint_exists(self, checkpoint_uri: str) -> bool:
        """Check if a checkpoint already exists.

        Args:
            checkpoint_uri: URI to check.

        Returns:
            True if checkpoint exists.
        """
        return self._storage.exists(checkpoint_uri)

    def discard_checkpoint(self, checkpoint_uri: str) -> None:
        """Discard an existing checkpoint.

        Args:
            checkpoint_uri: URI to discard.
        """
        self._storage.delete(checkpoint_uri)


class CompositeExperimentPersister:
    """Composes multiple persisters for flexible persistence strategies.

    This allows combining different persistence backends or adding
    additional persistence behaviors (e.g., logging, metrics tracking).
    """

    def __init__(self, persisters: list[ParquetExperimentPersister]) -> None:
        """Initialize with a list of persisters.

        Args:
            persisters: List of persisters to delegate to.
        """
        self._persisters = persisters

    def save_checkpoint(
        self,
        metrics: dict[str, Any],
        checkpoint_uri: str,
    ) -> None:
        """Save checkpoint using all persisters."""
        for persister in self._persisters:
            persister.save_checkpoint(metrics, checkpoint_uri)

    def save_model(
        self,
        model: BaseEstimator,
        context: ExperimentContext,
    ) -> None:
        """Save model using all persisters."""
        for persister in self._persisters:
            persister.save_model(model, context)

    def checkpoint_exists(self, checkpoint_uri: str) -> bool:
        """Check if checkpoint exists (uses first persister)."""
        if self._persisters:
            return self._persisters[0].checkpoint_exists(checkpoint_uri)
        return False

    def discard_checkpoint(self, checkpoint_uri: str) -> None:
        """Discard checkpoint using all persisters."""
        for persister in self._persisters:
            persister.discard_checkpoint(checkpoint_uri)


__all__ = [
    "ParquetExperimentPersister",
    "CompositeExperimentPersister",
]
