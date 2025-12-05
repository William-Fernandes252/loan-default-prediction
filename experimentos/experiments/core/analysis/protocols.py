"""Protocol definitions for analysis pipeline components.

This module defines the interfaces (protocols) for the three main stages
of the analysis pipeline: data loading, transformation, and export.
Using protocols enables dependency inversion and makes components
easily testable and replaceable.
"""

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import pandas as pd

from experiments.core.data import Dataset

# Type alias for translation functions (gettext-style)
TranslationFunc = Callable[[str], str]


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading experimental data.

    Implementations are responsible for loading data from a source
    (e.g., parquet files, databases) and returning a pandas DataFrame.
    """

    def load(self, dataset: Dataset) -> pd.DataFrame:
        """Load data for the given dataset.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A DataFrame containing the experimental results.
            Returns an empty DataFrame if no data is found.
        """
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for transforming loaded data.

    Implementations apply specific transformations to prepare data
    for visualization or export (e.g., aggregation, filtering, pivoting).
    """

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform the input DataFrame.

        Args:
            df: The input DataFrame to transform.
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

    Implementations handle the final output of the pipeline,
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
