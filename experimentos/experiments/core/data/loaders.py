"""Data loaders for the data processing pipeline.

This module provides implementations of the RawDataLoader protocol
for loading raw data from various sources (CSV, Parquet, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from experiments.core.data.protocols import RawDataPathProvider

if TYPE_CHECKING:
    from experiments.core.data import Dataset


class CsvRawDataLoader:
    """Loads raw data from CSV files.

    This loader reads CSV files from the raw data directory using
    the path provider to resolve dataset identifiers to file paths.

    Example:
        ```python
        loader = CsvRawDataLoader(context)  # Context implements RawDataPathProvider
        df = loader.load(Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(self, path_provider: RawDataPathProvider) -> None:
        """Initialize the loader.

        Args:
            path_provider: Provider for raw data file paths.
                Must implement the RawDataPathProvider protocol.
        """
        self._path_provider = path_provider

    def load(self, dataset: Dataset) -> pl.DataFrame:
        """Load raw data for the given dataset from CSV.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A Polars DataFrame containing the raw data.

        Raises:
            FileNotFoundError: If the raw data file does not exist.
        """
        path = self._path_provider.get_raw_data_path(dataset.id)

        # Base read options for all datasets
        read_options: dict[str, Any] = {
            "low_memory": False,
            "use_pyarrow": True,
        }

        # Add dataset-specific parameters (e.g., schema overrides)
        read_options.update(dataset.get_extra_params())

        return pl.read_csv(path, **read_options)
