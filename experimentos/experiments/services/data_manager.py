"""Service for managing experiment data, memory mapping, and result artifacts."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from experiments.core.data import Dataset

if TYPE_CHECKING:
    from experiments.services.storage_manager import StorageManager


class ExperimentDataManager:
    """
    Manages the lifecycle of data for experiments.

    Responsibilities:
    - Verifying existence of feature artifacts.
    - Creating efficient memory-mapped views of data for parallel processing.
    - Consolidating distributed checkpoint results into final artifacts.

    This class is a thin wrapper around StorageManager that implements
    the DataProvider protocol with URI-based operations.
    """

    def __init__(self, storage_manager: "StorageManager"):
        """Initialize with a storage manager.

        Args:
            storage_manager: Storage manager for all file operations.
        """
        self._storage_manager = storage_manager

    def artifacts_exist(self, dataset: Dataset) -> bool:
        """Checks if the necessary feature engineering artifacts exist."""
        return self._storage_manager.artifacts_exist(dataset)

    def get_dataset_size_gb(self, dataset: Dataset) -> float:
        """Estimates the size of the raw dataset in GB for job calculation."""
        return self._storage_manager.get_dataset_size_gb(dataset)

    @contextmanager
    def feature_context(self, dataset: Dataset) -> Generator[tuple[str, str], None, None]:
        """
        Context manager that prepares data for parallel access.

        Delegates to StorageManager's feature_context which handles:
        1. Loads Parquet files into memory.
        2. Dumps them to a temporary memory-mapped file (joblib format).
        3. Yields the paths to these memory maps.
        4. Cleans up temporary files and forces garbage collection on exit.

        Args:
            dataset: The dataset to load.

        Yields:
            tuple[str, str]: Paths to (X_mmap, y_mmap).

        Raises:
            FileNotFoundError: If feature artifacts don't exist.
        """
        with self._storage_manager.feature_context(dataset) as paths:
            yield paths

    def consolidate_results(self, dataset: Dataset) -> str | None:
        """
        Aggregates individual task checkpoints into a single results file.

        Delegates to StorageManager's consolidate_checkpoints method.

        Args:
            dataset: The dataset to consolidate results for.

        Returns:
            URI to the consolidated results file, or None if no checkpoints.
        """
        return self._storage_manager.consolidate_checkpoints(dataset.id)
