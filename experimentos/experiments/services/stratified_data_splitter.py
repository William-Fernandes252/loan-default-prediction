"""Implementation of stratified data splitting with validation."""

from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.core.training.data import TrainingData
from experiments.core.training.splitters import SplitData


class StratifiedDataSplitter:
    """Splits data using stratified train/test split with validation.

    This implementation:
    - Accepts Polars LazyFrames for deferred execution
    - Validates class distribution efficiently before full materialization
    - Performs stratified splitting using Scikit-Learn
    - Returns None if data validation fails (e.g., insufficient class samples)
    """

    def __init__(self, test_size: float = 0.30, cv_folds: int = 5) -> None:
        """Initialize the splitter.

        Args:
            test_size: Fraction of data to use for testing.
            cv_folds: Minimum number of samples required per class to allow
                      stratification (usually equals the number of CV folds).
        """
        self._test_size = test_size
        self._cv_folds = cv_folds

    def split(
        self,
        data: TrainingData,
        seed: int,
    ) -> SplitData:
        """Split data into train and test sets with validation.

        This method materializes the LazyFrames into Numpy arrays to perform
        the split via Scikit-Learn.

        Args:
            data: The training data to be split.
            seed: Random seed for reproducibility.

        Returns:
            SplitData containing numpy arrays if successful,
            None if validation fails.
        """
        X, y = data
        # 1. Validate: Check minimum class counts using Lazy API
        # We assume the target column is the first/only column in 'y'
        target_col, *_ = y.collect_schema().names()

        # Calculate class counts without loading full data yet
        class_counts_df = y.group_by(target_col).len().collect()

        min_class_count: int = cast(int, class_counts_df["len"].min())

        # Fail if any class has fewer than 2 samples (cannot split at all)
        if min_class_count < 2:
            raise Exception("Each class must have at least 2 samples for splitting.")

        # 2. Materialize Data for Splitting
        # Scikit-Learn requires in-memory arrays for stratification logic
        X_arr = X.collect().to_numpy()
        y_arr = y.collect().to_numpy().ravel()  # Flatten to 1D array

        # 3. Determine Stratification
        # Only stratify if every class has enough samples for the subsequent CV folds
        stratify_y = y_arr if min_class_count >= self._cv_folds else None

        # 4. Perform Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=self._test_size,
            stratify=stratify_y,
            random_state=seed,
        )

        # 5. Validate Training Split Sufficiency
        # Ensure the training set still has all classes
        if len(np.unique(y_train)) < 2:
            raise Exception("Training set must have at least 2 classes after split.")

        # Ensure the training set has enough samples for CV
        _, train_counts = np.unique(y_train, return_counts=True)
        if train_counts.min() < self._cv_folds:
            raise Exception("Each class in training set must have enough samples for CV.")

        return SplitData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )


__all__ = ["StratifiedDataSplitter"]
