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
    def it_returns_split_data_instance(
        self, stratified_splitter: StratifiedDataSplitter, balanced_training_data: TrainingData
    ) -> None:
        result = stratified_splitter.split(balanced_training_data, seed=42)

        assert result is not None
        assert isinstance(result, SplitData)

    def it_splits_according_to_test_size(
        self, stratified_splitter: StratifiedDataSplitter, balanced_training_data: TrainingData
    ) -> None:
        result = stratified_splitter.split(balanced_training_data, seed=42)

        assert result is not None
        total = len(result.X_train) + len(result.X_test)
        test_ratio = len(result.X_test) / total
        assert 0.25 <= test_ratio <= 0.35

    def it_produces_numpy_arrays(
        self, stratified_splitter: StratifiedDataSplitter, balanced_training_data: TrainingData
    ) -> None:
        result = stratified_splitter.split(balanced_training_data, seed=42)

        assert result is not None
        assert isinstance(result.X_train, np.ndarray)
        assert isinstance(result.X_test, np.ndarray)
        assert isinstance(result.y_train, np.ndarray)
        assert isinstance(result.y_test, np.ndarray)

    def it_preserves_both_classes_in_splits(
        self, stratified_splitter: StratifiedDataSplitter, balanced_training_data: TrainingData
    ) -> None:
        result = stratified_splitter.split(balanced_training_data, seed=42)

        assert result is not None
        assert len(np.unique(result.y_train)) == 2
        assert len(np.unique(result.y_test)) == 2

    def it_is_deterministic_with_same_seed(
        self, stratified_splitter: StratifiedDataSplitter, balanced_training_data: TrainingData
    ) -> None:
        result1 = stratified_splitter.split(balanced_training_data, seed=123)
        result2 = stratified_splitter.split(balanced_training_data, seed=123)

        assert result1 is not None and result2 is not None
        np.testing.assert_array_equal(result1.X_train, result2.X_train)
        np.testing.assert_array_equal(result1.y_train, result2.y_train)

    def it_produces_different_splits_with_different_seeds(
        self, stratified_splitter: StratifiedDataSplitter, balanced_training_data: TrainingData
    ) -> None:
        result1 = stratified_splitter.split(balanced_training_data, seed=1)
        result2 = stratified_splitter.split(balanced_training_data, seed=2)

        assert result1 is not None and result2 is not None
        assert not np.array_equal(result1.y_test, result2.y_test)

    def it_raises_when_class_has_too_few_samples(
        self, stratified_splitter: StratifiedDataSplitter
    ) -> None:
        X = pl.LazyFrame({"feature": [1, 2, 3]})
        y = pl.LazyFrame({"target": [0, 0, 1]})  # Only 1 sample of class 1
        data = TrainingData(X=X, y=y)

        with pytest.raises(Exception):
            stratified_splitter.split(data, seed=42)

    def it_handles_imbalanced_data(
        self, stratified_splitter: StratifiedDataSplitter, imbalanced_training_data: TrainingData
    ) -> None:
        result = stratified_splitter.split(imbalanced_training_data, seed=42)

        assert result is not None
        assert len(np.unique(result.y_train)) == 2


class DescribeSplitWithMinimalData:
    def it_raises_when_cv_folds_exceed_class_samples(self) -> None:
        splitter = StratifiedDataSplitter(test_size=0.30, cv_folds=5)
        X = pl.LazyFrame({"feature": list(range(10))})
        y = pl.LazyFrame({"target": [0] * 5 + [1] * 5})
        data = TrainingData(X=X, y=y)

        with pytest.raises(Exception):
            splitter.split(data, seed=42)
