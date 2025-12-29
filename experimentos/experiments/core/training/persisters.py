"""Results persistence implementations for the training pipeline."""

from typing import Protocol, runtime_checkable

from loguru import logger
import polars as pl

from experiments.core.data import Dataset
from experiments.core.training.protocols import (
    CheckpointUriProvider,
    ConsolidatedResultsUriProvider,
)
from experiments.services.storage import StorageService


@runtime_checkable
class ConsolidationUriProvider(CheckpointUriProvider, ConsolidatedResultsUriProvider, Protocol):
    """Protocol for providing consolidation-related URIs.

    This protocol combines checkpoint URI and consolidated results URI providers
    for backward compatibility. New code should use the more specific protocols:
    - CheckpointUriProvider for checkpoint URIs
    - ConsolidatedResultsUriProvider for consolidated results URIs
    """


class ParquetCheckpointPersister:
    """Persists experiment results as parquet checkpoint files using storage layer.

    This class handles the consolidation of individual checkpoint files
    into a single results file.
    """

    def __init__(
        self,
        storage: StorageService,
        checkpoint_uri_provider: CheckpointUriProvider,
        results_uri_provider: ConsolidatedResultsUriProvider,
    ) -> None:
        """Initialize the persister.

        Args:
            storage: Storage service for file operations.
            checkpoint_uri_provider: Provider for checkpoint URIs.
            results_uri_provider: Provider for consolidated results URIs.
        """
        self._storage = storage
        self._checkpoint_uri_provider = checkpoint_uri_provider
        self._results_uri_provider = results_uri_provider

    def _get_checkpoint_dir_uri(self, dataset: Dataset) -> str:
        """Get the checkpoint directory URI for a dataset."""
        # Use a sample checkpoint URI to find the parent directory
        sample_ckpt = self._checkpoint_uri_provider.get_checkpoint_uri(dataset.id, "x", "y", 0)
        # Extract parent directory from URI
        return "/".join(sample_ckpt.split("/")[:-1])

    def consolidate(self, dataset: Dataset) -> str | None:
        """Consolidate checkpoint results for a dataset.

        Args:
            dataset: The dataset to consolidate results for.

        Returns:
            URI to the consolidated results file, or None if no results.
        """
        ckpt_dir_uri = self._get_checkpoint_dir_uri(dataset)
        all_files = self._storage.list_files(ckpt_dir_uri, "*.parquet")

        if not all_files:
            logger.warning(f"No results found for {dataset.display_name}")
            return None

        logger.info(f"Consolidating {len(all_files)} results for {dataset.display_name}...")

        frames: list[pl.DataFrame] = []
        for uri in all_files:
            try:
                frames.append(self._storage.read_parquet(uri))
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {uri}: {e}")

        if not frames:
            logger.warning("No valid data frames could be loaded.")
            return None

        df_final = pl.concat(frames)
        final_output = self._results_uri_provider.get_consolidated_results_uri(dataset.id)

        self._storage.write_parquet(df_final, final_output)
        logger.success(f"Saved consolidated results to {final_output}")

        return final_output


# Backwards compatibility aliases (deprecated)
CheckpointPathProvider = CheckpointUriProvider
ConsolidatedResultsPathProvider = ConsolidatedResultsUriProvider
ConsolidationPathProvider = ConsolidationUriProvider


__all__ = [
    "CheckpointUriProvider",
    "ConsolidatedResultsUriProvider",
    "ConsolidationUriProvider",
    "ParquetCheckpointPersister",
    # Deprecated aliases
    "CheckpointPathProvider",
    "ConsolidatedResultsPathProvider",
    "ConsolidationPathProvider",
]
