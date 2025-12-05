"""Experiment persistence implementations for the experiment pipeline."""

from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
from sklearn.base import BaseEstimator

from experiments.core.experiment.protocols import ExperimentContext
from experiments.services.models import ModelVersioningService


class ParquetExperimentPersister:
    """Persists experiment results to parquet files and models via versioning service.

    This implementation:
    - Saves metrics to parquet checkpoint files
    - Uses ModelVersioningService for model persistence
    - Manages checkpoint lifecycle (exists, discard)
    """

    def __init__(
        self,
        model_versioning_service: ModelVersioningService | None = None,
    ) -> None:
        """Initialize the persister.

        Args:
            model_versioning_service: Optional service for model versioning.
        """
        self._model_versioning_service = model_versioning_service

    def save_checkpoint(
        self,
        metrics: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Save experiment results to a checkpoint file.

        Args:
            metrics: Metrics dictionary to save.
            checkpoint_path: Path to save the checkpoint.
        """
        pd.DataFrame([metrics]).to_parquet(checkpoint_path)

    def save_model(
        self,
        model: BaseEstimator,
        context: ExperimentContext,
    ) -> None:
        """Save a trained model using the versioning service.

        Args:
            model: The trained model to save.
            context: Experiment context (unused but available for metadata).
        """
        if self._model_versioning_service is None:
            return

        try:
            self._model_versioning_service.save_model(model, None)
        except Exception as e:
            logger.warning(f"Model save failed: {e}")

    def checkpoint_exists(self, checkpoint_path: Path) -> bool:
        """Check if a checkpoint already exists.

        Args:
            checkpoint_path: Path to check.

        Returns:
            True if checkpoint exists.
        """
        return checkpoint_path.exists()

    def discard_checkpoint(self, checkpoint_path: Path) -> None:
        """Discard an existing checkpoint.

        Args:
            checkpoint_path: Path to discard.
        """
        checkpoint_path.unlink(missing_ok=True)


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
        checkpoint_path: Path,
    ) -> None:
        """Save checkpoint using all persisters."""
        for persister in self._persisters:
            persister.save_checkpoint(metrics, checkpoint_path)

    def save_model(
        self,
        model: BaseEstimator,
        context: ExperimentContext,
    ) -> None:
        """Save model using all persisters."""
        for persister in self._persisters:
            persister.save_model(model, context)

    def checkpoint_exists(self, checkpoint_path: Path) -> bool:
        """Check if checkpoint exists (uses first persister)."""
        if self._persisters:
            return self._persisters[0].checkpoint_exists(checkpoint_path)
        return False

    def discard_checkpoint(self, checkpoint_path: Path) -> None:
        """Discard checkpoint using all persisters."""
        for persister in self._persisters:
            persister.discard_checkpoint(checkpoint_path)


__all__ = [
    "ParquetExperimentPersister",
    "CompositeExperimentPersister",
]
