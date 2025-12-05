"""Data loader implementations for the analysis pipeline.

This module provides concrete implementations of the DataLoader protocol
for loading experimental results from various sources.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from experiments.core.analysis.protocols import DataLoader, TranslationFunc
from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique


@runtime_checkable
class ResultsPathProvider(Protocol):
    """Protocol for providing paths to consolidated results."""

    def get_latest_consolidated_results_path(self, dataset_id: str) -> Path | None:
        """Get the path to the latest consolidated results for a dataset."""
        ...


def _get_model_display(model_id: str, translate: TranslationFunc) -> str:
    """Returns the translated display name for a model."""
    try:
        return translate(ModelType.from_id(model_id).display_name)
    except ValueError:
        return model_id


def _get_technique_display(technique_id: str, translate: TranslationFunc) -> str:
    """Returns the translated display name for a technique."""
    try:
        return translate(Technique.display_name_from_id(technique_id))
    except ValueError:
        return technique_id


class ParquetResultsLoader:
    """Loads experimental results from parquet files.

    This loader reads consolidated results from parquet files.

    Attributes:
        path_provider: Provider for results file paths.
    """

    def __init__(
        self,
        path_provider: ResultsPathProvider,
    ) -> None:
        """Initialize the loader.

        Args:
            path_provider: Object providing paths to results files.
        """
        self._path_provider = path_provider

    def load(self, dataset: Dataset) -> pd.DataFrame:
        """Load data for the given dataset.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A DataFrame containing the experimental results.
            Returns an empty DataFrame if no data is found.
        """
        path = self._path_provider.get_latest_consolidated_results_path(dataset.id)
        if path is None or not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
            return df
        except Exception:
            # Return empty DataFrame if file is corrupted or unreadable
            return pd.DataFrame()


class DisplayColumnEnricher:
    """Enriches data with display columns for visualization.

    This enricher adds translated display columns for models and techniques,
    making the data ready for presentation and visualization.

    Attributes:
        translate: Translation function for display names.
    """

    def __init__(self, translate: TranslationFunc) -> None:
        """Initialize the enricher.

        Args:
            translate: Translation function for display names.
        """
        self._translate = translate

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add display columns for models and techniques.

        Args:
            df: The input DataFrame to enrich.

        Returns:
            A DataFrame with added display columns for model and technique names.
            Returns the original DataFrame if it's empty.
        """
        if df.empty:
            return df

        df = df.copy()

        # Apply translations to model and technique columns
        if "model" in df.columns:
            df["model_display"] = df["model"].apply(
                lambda x: _get_model_display(x, self._translate)
            )
        if "technique" in df.columns:
            df["technique_display"] = df["technique"].apply(
                lambda x: _get_technique_display(x, self._translate)
            )

        return df


class EnrichedResultsLoader:
    """Composite loader that loads data and enriches it with display columns.

    This loader combines the responsibilities of loading raw data and
    enriching it with display columns, providing a convenient interface
    for the analysis pipeline.

    Attributes:
        loader: The underlying data loader.
        enricher: The display column enricher.
    """

    def __init__(
        self,
        path_provider: ResultsPathProvider,
        translate: TranslationFunc,
    ) -> None:
        """Initialize the composite loader.

        Args:
            path_provider: Provider for results file paths.
            translate: Translation function for display names.
        """
        self._loader = ParquetResultsLoader(path_provider)
        self._enricher = DisplayColumnEnricher(translate)

    def load(self, dataset: Dataset) -> pd.DataFrame:
        """Load and enrich data for the given dataset.

        Args:
            dataset: The dataset to load data for.

        Returns:
            A DataFrame containing the experimental results with
            added display columns for model and technique names.
            Returns an empty DataFrame if no data is found.
        """
        df = self._loader.load(dataset)
        return self._enricher.enrich(df)


# Re-export the protocol for convenience
__all__ = [
    "DataLoader",
    "DisplayColumnEnricher",
    "EnrichedResultsLoader",
    "ParquetResultsLoader",
    "ResultsPathProvider",
]
