"""Tests for stratified_data_splitter service."""

import numpy as np
import polars as pl
import pytest

from experiments.core.training.data import TrainingData
from experiments.core.training.splitters import SplitData
from experiments.services.stratified_data_splitter import StratifiedDataSplitter


class DescribeStratifiedDataSplitterInit:
    def it_uses_default_test_size(self) -> None:
        splitter = StratifiedDataSplitter()

        assert splitter._test_size == 0.30

    def it_uses_default_cv_folds(self) -> None:
        splitter = StratifiedDataSplitter()

        assert splitter._cv_folds == 5

    def it_accepts_custom_test_size(self) -> None:
        splitter = StratifiedDataSplitter(test_size=0.20)

        assert splitter._test_size == 0.20

    def it_accepts_custom_cv_folds(self) -> None:
        splitter = StratifiedDataSplitter(cv_folds=10)

        assert splitter._cv_folds == 10


class DescribeSplit:
    @pytest.fixture
    def splitter(self) -> StratifiedDataSplitter:
        return StratifiedDataSplitter(test_size=0.30, cv_folds=2)

    @pytest.fixture
    def balanced_data(self) -> TrainingData:
        """Creates balanced training data with 50 samples per class."""
        n_samples = 100
        X = pl.LazyFrame(
            {
                "feature1": list(range(n_samples)),
                "feature2": list(range(n_samples, n_samples * 2)),
            }
        )
        y = pl.LazyFrame({"target": [0] * 50 + [1] * 50})
        return TrainingData(X=X, y=y)

    def it_returns_split_data(
        self, splitter: StratifiedDataSplitter, balanced_data: TrainingData
    ) -> None:
        result = splitter.split(balanced_data, seed=42)

        assert result is not None
        assert isinstance(result, SplitData)

    def it_splits_data_according_to_test_size(
        self, splitter: StratifiedDataSplitter, balanced_data: TrainingData
    ) -> None:
        result = splitter.split(balanced_data, seed=42)

        assert result is not None
        total = len(result.X_train) + len(result.X_test)
        test_ratio = len(result.X_test) / total
        assert 0.25 <= test_ratio <= 0.35  # Allow some tolerance

    def it_produces_numpy_arrays(
        self, splitter: StratifiedDataSplitter, balanced_data: TrainingData
    ) -> None:
        result = splitter.split(balanced_data, seed=42)

        assert result is not None
        assert isinstance(result.X_train, np.ndarray)
        assert isinstance(result.X_test, np.ndarray)
        assert isinstance(result.y_train, np.ndarray)
        assert isinstance(result.y_test, np.ndarray)

    def it_preserves_both_classes_in_train_and_test(
        self, splitter: StratifiedDataSplitter, balanced_data: TrainingData
    ) -> None:
        result = splitter.split(balanced_data, seed=42)

        assert result is not None
        assert len(np.unique(result.y_train)) == 2
        assert len(np.unique(result.y_test)) == 2

    def it_produces_deterministic_splits_with_same_seed(
        self, splitter: StratifiedDataSplitter, balanced_data: TrainingData
    ) -> None:
        result1 = splitter.split(balanced_data, seed=123)
        result2 = splitter.split(balanced_data, seed=123)

        assert result1 is not None
        assert result2 is not None
        np.testing.assert_array_equal(result1.X_train, result2.X_train)
        np.testing.assert_array_equal(result1.y_train, result2.y_train)

    def it_produces_different_splits_with_different_seeds(
        self, splitter: StratifiedDataSplitter, balanced_data: TrainingData
    ) -> None:
        result1 = splitter.split(balanced_data, seed=1)
        result2 = splitter.split(balanced_data, seed=2)

        assert result1 is not None
        assert result2 is not None
        # The splits should differ (at least y_test should be different)
        assert not np.array_equal(result1.y_test, result2.y_test)

    def it_raises_when_class_has_fewer_than_two_samples(
        self, splitter: StratifiedDataSplitter
    ) -> None:
        # One class with only 1 sample - cannot split
        X = pl.LazyFrame({"feature": [1, 2, 3]})
        y = pl.LazyFrame({"target": [0, 0, 1]})  # Only 1 sample of class 1
        data = TrainingData(X=X, y=y)

        # The splitter should handle edge cases with few samples
        # Depending on implementation, this may raise or return None
        with pytest.raises(Exception):
            splitter.split(data, seed=42)

    def it_handles_imbalanced_data(self, splitter: StratifiedDataSplitter) -> None:
        # Highly imbalanced: 80% class 0, 20% class 1
        n_samples = 50
        X = pl.LazyFrame({"feature": list(range(n_samples))})
        y = pl.LazyFrame({"target": [0] * 40 + [1] * 10})
        data = TrainingData(X=X, y=y)

        result = splitter.split(data, seed=42)

        # Should still produce a valid split
        assert result is not None
        assert len(np.unique(result.y_train)) == 2


class DescribeSplitWithMinimalData:
    def it_raises_when_training_has_insufficient_samples_for_cv(self) -> None:
        # With cv_folds=5, each class needs at least 5 samples in training
        splitter = StratifiedDataSplitter(test_size=0.30, cv_folds=5)

        # Only 10 samples total, 5 per class. After 30% test split, ~3-4 per class in train
        X = pl.LazyFrame({"feature": list(range(10))})
        y = pl.LazyFrame({"target": [0] * 5 + [1] * 5})
        data = TrainingData(X=X, y=y)

        with pytest.raises(Exception):
            splitter.split(data, seed=42)
