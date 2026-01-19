"""Definitions of feature extraction and target separation utilities."""

from typing import NamedTuple, Protocol

import polars as pl


class TrainingDataset(NamedTuple):
    """Represents a dataset ready for training, with features and target separated."""

    X: pl.DataFrame
    y: pl.DataFrame


class FeatureExtractor(Protocol):
    """Protocol for feature extraction functions."""

    def extract_features_and_target(self, data: pl.DataFrame) -> TrainingDataset:
        """Extracts features from the given data frame.

        Args:
            data: The input `DataFrame`.

        Returns:
            A `TrainingDataset` containing the extracted features and target.
        """
        ...
