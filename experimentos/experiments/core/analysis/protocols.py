"""Protocol definitions for analysis pipeline components.

This module defines the interfaces (protocols) for the three main stages
of the analysis pipeline: data loading, transformation, and export.

The analysis pipeline uses pandas DataFrames (for compatibility with
visualization libraries), unlike the data processing pipeline which
uses Polars DataFrames for better performance with large datasets.

Note:
    The base load/transform/export pattern is defined in
    `experiments.core.data.protocols`. This module provides
    analysis-specific protocol variants that work with pandas.
"""

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import pandas as pd

from experiments.core.data import Dataset

# Type alias for translation functions (gettext-style)
TranslationFunc = Callable[[str], str]


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading experimental data for analysis.

    Unlike RawDataLoader in data.protocols (which loads raw CSV data),
    this loader is designed for loading experiment results from
    consolidated Parquet files for analysis and visualization.
    """

    def load(self, dataset: Dataset) -> pd.DataFrame:
        """Load analysis data for the given dataset.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A pandas DataFrame containing the experimental results.
            Returns an empty DataFrame if no data is found.
        """
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for transforming loaded analysis data.

    Unlike DataTransformer in data.protocols (which processes raw data),
    this transformer prepares data for visualization or export
    (e.g., aggregation, filtering, pivoting for charts).
    """

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform the input DataFrame for analysis output.

        Args:
            df: The input pandas DataFrame to transform.
            dataset: The dataset being analyzed (for context).

        Returns:
            A dictionary containing:
                - 'data': The transformed DataFrame(s)
                - Additional metadata needed for export (e.g., plot configs)
        """
        ...


@runtime_checkable
class DataExporter(Protocol):
    """Protocol for exporting analysis results.

    Implementations handle the final output of the analysis pipeline,
    whether to files (PNG, CSV, LaTeX) or other destinations.
    """

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export the analysis results.

        Args:
            data: The transformed data dictionary from a transformer.
            output_dir: The directory to save outputs to.
            dataset: The dataset being analyzed.

        Returns:
            A list of paths to the exported files.
        """
        ...
