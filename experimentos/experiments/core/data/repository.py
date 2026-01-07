"""Defines the data repository protocol.

It specifies the expected interface for data repositories used in the experiment datasets.
"""

from typing import Protocol

import polars as pl

from experiments.core.data import Dataset


class DataRepository(Protocol):
    """Protocol for data repositories."""

    def get_size_in_bytes(self, dataset: Dataset) -> int:
        """Gets the size in bytes of the raw data for the specified dataset.

        Args:
            dataset (Dataset): The dataset for which to get the size.

        Returns:
            int: The size of the raw data in bytes.

        Raises:
            Exception: If there is an error retrieving the size.
        """
        ...

    def get_raw_data(self, dataset: Dataset) -> pl.LazyFrame:
        """Fetches the raw data for the specified dataset.

        Args:
            dataset (Dataset): The dataset for which to fetch raw data.

        Returns:
            pl.LazyFrame: The raw data as a Polars LazyFrame.

        Raises:
            Exception: If there is an error retrieving the data.
        """
        ...

    def save_interim_data(self, dataset: Dataset, data: pl.DataFrame) -> pl.LazyFrame:
        """Saves the transformed (ready to split) data for the specified dataset.

        Args:
            dataset (Dataset): The dataset for which to fetch processed data.
            data (pl.LazyFrame): The processed data to save.
        Returns:
            pl.LazyFrame: The processed data as a Polars LazyFrame.

        Raises:
            Exception: If there is an error retrieving the data.
        """
        ...

    def get_interim_data(self, dataset: Dataset) -> pl.LazyFrame:
        """Fetches the transformed (ready to split) data for the specified dataset.

        Args:
            dataset (Dataset): The dataset for which to fetch processed data.

        Returns:
            pl.LazyFrame: The processed data as a Polars LazyFrame.

        Raises:
            Exception: If there is an error retrieving the data.
        """
        ...

    def save_final_features(self, dataset: Dataset, X: pl.DataFrame, y: pl.DataFrame) -> None:
        """Saves the final features and target for the specified dataset.

        Args:
            dataset (Dataset): The dataset for which to save final features.
            X (pl.DataFrame): The final features data.
            y (pl.DataFrame): The final target data.

        Raises:
            Exception: If there is an error saving the data.
        """
        ...

    def is_processed(self, dataset: Dataset) -> bool:
        """Checks if the processed data for the specified dataset exists.

        Args:
            dataset (Dataset): The dataset to check.

        Returns:
            bool: True if processed data exists, False otherwise.

        Raises:
            Exception: If there is an error checking the data.
        """
        ...
