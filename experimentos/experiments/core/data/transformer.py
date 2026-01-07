"""Base class for data transformers.

This module provides the abstract base class for dataset-specific
data transformers that implement the DataTransformer protocol.
"""

from __future__ import annotations

from typing import Callable, Literal

import polars as pl

_PolarsEngine = Literal["auto", "gpu"]
"""Type alias for Polars execution engines."""


type Transformer = Callable[[pl.DataFrame | pl.LazyFrame, bool], pl.DataFrame]
"""A transformer receives the raw data from the dataset and do cleaning, feature engineering, and other transformations.

Args:
    df: The raw Polars DataFrame or LazyFrame from the dataset.
    use_gpu: Whether to enable GPU acceleration for transformations.

Returns:
    A cleaned Polars DataFrame ready for the ML pipeline.
"""


def get_engine(use_gpu: bool) -> _PolarsEngine:
    """Get the appropriate Polars execution engine based on GPU usage.

    Args:
        use_gpu: Whether to enable GPU acceleration.

    Returns:
        The Polars execution engine to use.
    """
    return "gpu" if use_gpu else "auto"
