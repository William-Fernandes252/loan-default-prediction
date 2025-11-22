"""Base class for data processors."""

from abc import ABC, abstractmethod
from typing import Literal

import polars as pl

_PolarsEngine = Literal["auto", "gpu"]
"""Type alias for Polars execution engines."""


class DataProcessor(ABC):
    """Abstract base class for dataset-specific data processors."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

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

    def _get_engine(self) -> _PolarsEngine:
        """Returns the appropriate execution engine based on GPU usage."""
        return "gpu" if self.use_gpu else "auto"
