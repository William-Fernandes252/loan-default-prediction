"""Definitions for data splitting protocols and results."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from experiments.core.training.data import TrainingData


@dataclass(slots=True)
class SplitData:
    """Result of data splitting.

    Attributes:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for splitting data into train/test sets."""

    def split(
        self,
        data: TrainingData,
        seed: int,
    ) -> SplitData:
        """Split data into train and test sets.

        Args:
            data (TrainingData): The training data to be split.
            seed (int): Random seed for reproducibility.

        Returns:
            SplitData if successful, None if validation fails.
        """
        ...
