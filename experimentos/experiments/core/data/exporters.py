"""Data exporters for the data processing pipeline.

This module provides implementations of the ProcessedDataExporter protocol
for persisting processed data to various storage formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from experiments.core.data.protocols import InterimDataPathProvider

if TYPE_CHECKING:
    from experiments.core.data import Dataset


class ParquetDataExporter:
    """Exports processed data to Parquet files.

    This exporter writes processed DataFrames to the interim data
    directory using the path provider to resolve output locations.

    Example:
        ```python
        exporter = ParquetDataExporter(context)  # Context implements InterimDataPathProvider
        path = exporter.export(processed_df, Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(self, path_provider: InterimDataPathProvider) -> None:
        """Initialize the exporter.

        Args:
            path_provider: Provider for interim data file paths.
                Must implement the InterimDataPathProvider protocol.
        """
        self._path_provider = path_provider

    def export(self, df: pl.DataFrame, dataset: Dataset) -> Path:
        """Export processed data to a Parquet file.

        Args:
            df: The processed DataFrame to export.
            dataset: The dataset being processed.

        Returns:
            The path to the exported Parquet file.
        """
        output_path = self._path_provider.get_interim_data_path(dataset.id)

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the DataFrame to Parquet
        df.write_parquet(output_path)

        return output_path
