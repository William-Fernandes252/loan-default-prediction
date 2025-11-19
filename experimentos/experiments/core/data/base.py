"""Base class for data processors."""

from abc import ABC, abstractmethod

import polars as pl


class DataProcessor(ABC):
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @abstractmethod
    def process(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        """Core transformation logic."""
        pass

    def sanitize(self, df: pl.DataFrame) -> pl.DataFrame:
        """Shared logic to handle infs/nans."""
        return df
