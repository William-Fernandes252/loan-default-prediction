"""Results persistence implementations for the training pipeline."""

from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from loguru import logger
import pandas as pd

from experiments.core.data import Dataset
from experiments.core.training.protocols import (
    CheckpointPathProvider,
    ConsolidatedResultsPathProvider,
)


@runtime_checkable
class ConsolidationPathProvider(CheckpointPathProvider, ConsolidatedResultsPathProvider, Protocol):
    """Protocol for providing consolidation-related paths.

    This protocol combines checkpoint path and consolidated results path providers
    for backward compatibility. New code should use the more specific protocols:
    - CheckpointPathProvider for checkpoint paths
    - ConsolidatedResultsPathProvider for consolidated results paths
    """


class ParquetCheckpointPersister:
    """Persists experiment results as parquet checkpoint files.

    This class handles the consolidation of individual checkpoint files
    into a single results file.
    """

    def __init__(
        self,
        checkpoint_path_provider: CheckpointPathProvider,
        results_path_provider: ConsolidatedResultsPathProvider,
    ) -> None:
        """Initialize the persister.

        Args:
            checkpoint_path_provider: Provider for checkpoint paths.
            results_path_provider: Provider for consolidated results paths.
        """
        self._checkpoint_path_provider = checkpoint_path_provider
        self._results_path_provider = results_path_provider

    def _get_checkpoint_dir(self, dataset: Dataset) -> Path:
        """Get the checkpoint directory for a dataset."""
        # Use a sample checkpoint path to find the parent directory
        sample_ckpt = self._checkpoint_path_provider.get_checkpoint_path(dataset.id, "x", "y", 0)
        return sample_ckpt.parent

    def consolidate(self, dataset: Dataset) -> Path | None:
        """Consolidate checkpoint results for a dataset.

        Args:
            dataset: The dataset to consolidate results for.

        Returns:
            Path to the consolidated results file, or None if no results.
        """
        ckpt_dir = self._get_checkpoint_dir(dataset)
        all_files = list(ckpt_dir.glob("*.parquet"))

        if not all_files:
            logger.warning(f"No results found for {dataset.display_name}")
            return None

        logger.info(f"Consolidating {len(all_files)} results for {dataset.display_name}...")

        frames: list[pd.DataFrame] = []
        for fp in all_files:
            try:
                frames.append(pd.read_parquet(fp))
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {fp.name}: {e}")

        if not frames:
            logger.warning("No valid data frames could be loaded.")
            return None

        df_final = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
        final_output = self._results_path_provider.get_consolidated_results_path(dataset.id)

        # Ensure results directory exists
        final_output.parent.mkdir(parents=True, exist_ok=True)

        df_final.to_parquet(final_output)
        logger.success(f"Saved consolidated results to {final_output}")

        return final_output


__all__ = [
    "CheckpointPathProvider",
    "ConsolidatedResultsPathProvider",
    "ConsolidationPathProvider",
    "ParquetCheckpointPersister",
]
