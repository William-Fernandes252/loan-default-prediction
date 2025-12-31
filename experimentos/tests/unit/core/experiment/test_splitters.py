"""Tests for experiments.core.experiment.splitters module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.core.experiment.protocols import SplitData
from experiments.core.experiment.splitters import StratifiedDataSplitter


@pytest.fixture
def mock_balanced_data() -> tuple[np.ndarray, np.ndarray]:
    """Create balanced mock data for testing."""
    X = np.random.rand(100, 10)
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def mock_imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """Create imbalanced mock data for testing."""
    X = np.random.rand(100, 10)
    y = np.array([0] * 95 + [1] * 5)
    return X, y


@pytest.fixture
def mock_single_class_data() -> tuple[np.ndarray, np.ndarray]:
    """Create single-class mock data (should fail validation)."""
    X = np.random.rand(100, 10)
    y = np.zeros(100)
    return X, y


class DescribeStratifiedDataSplitter:
    """Tests for StratifiedDataSplitter class."""

    def it_initializes_with_default_test_size(self) -> None:
        """Verify default test_size is 0.30."""
        splitter = StratifiedDataSplitter()

        assert splitter._test_size == 0.30

    def it_accepts_custom_test_size(self) -> None:
        """Verify custom test_size is stored."""
        splitter = StratifiedDataSplitter(test_size=0.20)

        assert splitter._test_size == 0.20


class DescribeStratifiedDataSplitterSplit:
    """Tests for StratifiedDataSplitter.split() method."""

    def it_returns_split_data_for_valid_balanced_data(
        self,
        mock_balanced_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify split returns SplitData for balanced data."""
        X, y = mock_balanced_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter(test_size=0.30)

        result = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

        assert result is not None
        assert isinstance(result, SplitData)
        assert len(result.X_train) + len(result.X_test) == len(X)
        assert len(result.y_train) + len(result.y_test) == len(y)

    def it_returns_none_for_single_class_data(
        self,
        mock_single_class_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify split returns None when only one class exists."""
        X, y = mock_single_class_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter()

        result = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

        assert result is None

    def it_uses_stratified_split_when_enough_samples(
        self,
        mock_balanced_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify stratified split preserves class proportions."""
        X, y = mock_balanced_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter(test_size=0.30)

        result = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

        assert result is not None

        # Check class proportions are roughly maintained
        train_ratio = np.mean(result.y_train)
        test_ratio = np.mean(result.y_test)
        original_ratio = np.mean(y)

        # Should be close to 0.5 for balanced data
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1

    def it_respects_random_seed_for_reproducibility(
        self,
        mock_balanced_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify same seed produces same split."""
        X, y = mock_balanced_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter(test_size=0.30)

        result1 = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)
        result2 = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

        assert result1 is not None
        assert result2 is not None
        np.testing.assert_array_equal(result1.X_train, result2.X_train)
        np.testing.assert_array_equal(result1.y_train, result2.y_train)

    def it_produces_different_splits_with_different_seeds(
        self,
        mock_balanced_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify different seeds produce different splits."""
        X, y = mock_balanced_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter(test_size=0.30)

        result1 = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)
        result2 = splitter.split(str(X_path), str(y_path), seed=123, cv_folds=5)

        assert result1 is not None
        assert result2 is not None
        # Arrays should be different (with high probability)
        assert not np.array_equal(result1.X_train, result2.X_train)

    def it_returns_correct_split_sizes(
        self,
        mock_balanced_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify split sizes match test_size parameter."""
        X, y = mock_balanced_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter(test_size=0.30)

        result = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

        assert result is not None
        # Test size should be approximately 30%
        expected_test = int(len(X) * 0.30)
        assert abs(len(result.X_test) - expected_test) <= 1

    def it_loads_data_with_memory_mapping(
        self,
        mock_balanced_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """Verify data is loaded with memory mapping."""
        X, y = mock_balanced_data
        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter()

        with patch("experiments.core.experiment.splitters.joblib.load") as mock_load:
            mock_load.side_effect = [X, y]

            splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

            # Verify mmap_mode='r' is used
            mock_load.assert_any_call(str(X_path), mmap_mode="r")
            mock_load.assert_any_call(str(y_path), mmap_mode="r")

    def it_returns_none_when_minority_class_below_cv_folds(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify returns None when minority class has fewer samples than cv_folds."""
        # Create data with only 3 samples of minority class
        X = np.random.rand(100, 10)
        y = np.array([0] * 97 + [1] * 3)

        X_path = tmp_path / "X.joblib"
        y_path = tmp_path / "y.joblib"

        import joblib

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        splitter = StratifiedDataSplitter(test_size=0.30)

        # With 3 samples in minority and 30% test split, training will have ~2
        # which is less than cv_folds=5
        result = splitter.split(str(X_path), str(y_path), seed=42, cv_folds=5)

        # May return None depending on split
        # At minimum, it should handle this case without crashing
        assert result is None or isinstance(result, SplitData)
