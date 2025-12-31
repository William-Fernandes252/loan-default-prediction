"""Protocol definitions for data processing pipeline components.

This module defines the interfaces (protocols) for the three main stages
of the data processing pipeline: loading, transformation, and export.
Using protocols enables dependency inversion and makes components
easily testable and replaceable.

These protocols are also used by the analysis pipeline, which follows
the same load → transform → export pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from experiments.core.data import Dataset


@runtime_checkable
class RawDataUriProvider(Protocol):
    """Protocol for providing URIs to raw data files.

    Implementations are responsible for resolving dataset identifiers
    to their corresponding raw data file URIs.
    """

    def get_raw_data_uri(self, dataset_id: str) -> str:
        """Get the URI to the raw data file for a dataset.

        Args:
            dataset_id: The dataset identifier (e.g., 'taiwan_credit').

        Returns:
            The URI to the raw data file.
        """
        ...


@runtime_checkable
class InterimDataUriProvider(Protocol):
    """Protocol for providing URIs to interim (processed) data files.

    Implementations are responsible for resolving dataset identifiers
    to their corresponding interim data file URIs.
    """

    def get_interim_data_uri(self, dataset_id: str) -> str:
        """Get the URI to the interim data file for a dataset.

        Args:
            dataset_id: The dataset identifier (e.g., 'taiwan_credit').

        Returns:
            The URI to the interim data file.
        """
        ...


@runtime_checkable
class RawDataLoader(Protocol):
    """Protocol for loading raw data.

    Implementations are responsible for loading data from a source
    (e.g., CSV files) and returning a Polars DataFrame or LazyFrame.
    """

    def load(self, dataset: Dataset) -> pl.DataFrame | pl.LazyFrame:
        """Load raw data for the given dataset.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A Polars DataFrame or LazyFrame containing the raw data.
        """
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for transforming loaded data.

    Implementations apply dataset-specific transformations to prepare
    data for modeling (e.g., feature engineering, encoding, cleaning).
    """

    def transform(
        self, df: pl.DataFrame | pl.LazyFrame, dataset: Dataset
    ) -> pl.DataFrame | pl.LazyFrame:
        """Transform the input DataFrame.

        Args:
            df: The input Polars DataFrame or LazyFrame to transform.
            dataset: The dataset being processed (for context).

        Returns:
            The transformed Polars DataFrame or LazyFrame.
        """
        ...


@runtime_checkable
class ProcessedDataExporter(Protocol):
    """Protocol for exporting processed data.

    Implementations handle persisting the transformed data
    to storage (e.g., Parquet files).
    """

    def export(self, df: pl.DataFrame, dataset: Dataset) -> str:
        """Export the processed data.

        Args:
            df: The transformed DataFrame to export.
            dataset: The dataset being processed.

        Returns:
            The URI to the exported file.
        """
        ...


# Backwards compatibility aliases (deprecated)
RawDataPathProvider = RawDataUriProvider
InterimDataPathProvider = InterimDataUriProvider
