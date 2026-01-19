"""Defines data structures and protocols for training data loading."""

from typing import NamedTuple, Protocol

import polars as pl

from experiments.core.data.datasets import Dataset


class TrainingData(NamedTuple):
    """Represents training data splits.

    Attributes:
        X (pl.LazyFrame): Features for training.
        y (pl.LazyFrame): Target labels for training.
    """

    X: pl.LazyFrame
    y: pl.LazyFrame


class TrainingDataLoader(Protocol):
    """Protocol for retrieving training data splits."""

    def load_training_data(self, dataset: Dataset) -> TrainingData:
        """Loads and returns the training data splits.

        Returns:
            A dictionary containing the training data splits.
        """
        ...
