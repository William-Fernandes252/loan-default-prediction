"""Base class for data transformers.

This module provides the abstract base class for dataset-specific
data transformers that implement the DataTransformer protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from experiments.core.data import Dataset

_PolarsEngine = Literal["auto", "gpu"]
"""Type alias for Polars execution engines."""


class BaseDataTransformer(ABC):
    """Abstract base class for dataset-specific data transformers.

    This class provides the foundation for implementing dataset-specific
    transformation logic. Subclasses must implement the `_apply_transformations`
    method with the actual transformation logic.

    The `transform` method orchestrates the transformation process:
    1. Apply dataset-specific transformations via `_apply_transformations`
    2. Apply shared sanitization via `sanitize`

    Attributes:
        use_gpu: Whether to use GPU acceleration for transformations.

    Example:
        ```python
        class MyDataTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "my_dataset"

            def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
                # Dataset-specific transformation logic
                return df.with_columns(...)
        ```
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialize the transformer.

        Args:
            use_gpu: Whether to use GPU acceleration for transformations.
        """
        self.use_gpu = use_gpu

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """The name of the dataset this transformer handles."""
        ...

    def transform(
        self, df: pl.DataFrame | pl.LazyFrame, dataset: Dataset
    ) -> pl.DataFrame | pl.LazyFrame:
        """Transform raw data into processed format.

        This method orchestrates the transformation pipeline:
        1. Apply dataset-specific transformations
        2. Apply shared sanitization

        Args:
            df: The raw input DataFrame or LazyFrame.
            dataset: The dataset being processed (for context/validation).

        Returns:
            The transformed DataFrame or LazyFrame ready for modeling.
        """
        transformed = self._apply_transformations(df)
        return self.sanitize(transformed)

    @abstractmethod
    def _apply_transformations(
        self, df: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        """Apply dataset-specific transformation logic.

        This method should contain all the feature engineering,
        encoding, and cleaning logic specific to the dataset.

        Args:
            df: The raw input DataFrame or LazyFrame.

        Returns:
            The transformed DataFrame or LazyFrame.
        """
        ...

    def sanitize(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Apply shared sanitization logic to handle edge cases.

        Override this method to add custom sanitization logic
        (e.g., handling infinities, nulls, etc.).

        Args:
            df: The DataFrame to sanitize.

        Returns:
            The sanitized DataFrame.
        """
        return df

    def _get_engine(self) -> _PolarsEngine:
        """Returns the appropriate execution engine based on GPU usage."""
        return "gpu" if self.use_gpu else "auto"


# Backward compatibility alias
DataProcessor = BaseDataTransformer
