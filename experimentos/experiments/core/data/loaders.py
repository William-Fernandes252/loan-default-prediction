"""Data loaders for the data processing pipeline.

This module provides implementations of the RawDataLoader protocol
for loading raw data from various sources (CSV, Parquet, etc.)
using the storage abstraction layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from experiments.core.data import Dataset
    from experiments.services.storage import StorageService


class CsvRawDataLoader:
    """Loads raw data from CSV files using the storage layer.

    This loader reads CSV files from the raw data directory using
    the storage service to handle file operations.

    Example:
        ```python
        loader = CsvRawDataLoader(storage_manager)
        df = loader.load(Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(self, storage: StorageService, base_uri: str) -> None:
        """Initialize the loader.

        Args:
            storage: Storage service for file operations.
            base_uri: Base URI for raw data files.
        """
        self._storage = storage
        self._base_uri = base_uri.rstrip("/")

    def _get_uri(self, dataset_id: str) -> str:
        """Get the URI for a dataset's raw CSV file."""
        return f"{self._base_uri}/{dataset_id}.csv"

    def load(self, dataset: Dataset) -> pl.DataFrame:
        """Load raw data for the given dataset from CSV.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A Polars DataFrame containing the raw data.

        Raises:
            FileDoesNotExistError: If the raw data file does not exist.
        """
        uri = self._get_uri(dataset.id)

        # Base read options for all datasets
        read_options: dict[str, Any] = {
            "low_memory": False,
            "use_pyarrow": True,
        }

        # Add dataset-specific parameters (e.g., schema overrides)
        read_options.update(dataset.get_extra_params())

        return self._storage.read_csv(uri, **read_options)
