"""Data exporters for the data processing pipeline.

This module provides implementations of the ProcessedDataExporter protocol
for persisting processed data to various storage formats using the
storage abstraction layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from experiments.core.data import Dataset
    from experiments.services.storage import StorageService


class ParquetDataExporter:
    """Exports processed data to Parquet files using the storage layer.

    This exporter writes processed DataFrames to the interim data
    directory using the storage service to handle file operations.

    Example:
        ```python
        exporter = ParquetDataExporter(storage, base_uri)
        uri = exporter.export(processed_df, Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(self, storage: StorageService, base_uri: str) -> None:
        """Initialize the exporter.

        Args:
            storage: Storage service for file operations.
            base_uri: Base URI for interim data files.
        """
        self._storage = storage
        self._base_uri = base_uri.rstrip("/")

    def _get_uri(self, dataset_id: str) -> str:
        """Get the URI for a dataset's interim parquet file."""
        return f"{self._base_uri}/{dataset_id}.parquet"

    def export(self, df: pl.DataFrame, dataset: Dataset) -> str:
        """Export processed data to a Parquet file.

        Args:
            df: The processed DataFrame to export.
            dataset: The dataset being processed.

        Returns:
            The URI to the exported Parquet file.
        """
        output_uri = self._get_uri(dataset.id)

        # Write the DataFrame to Parquet via storage layer
        self._storage.write_parquet(df, output_uri)

        return output_uri
