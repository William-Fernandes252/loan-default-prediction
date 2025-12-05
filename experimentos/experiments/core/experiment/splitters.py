"""Data splitting implementations for the experiment pipeline."""

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from experiments.core.experiment.protocols import SplitData


class StratifiedDataSplitter:
    """Splits data using stratified train/test split with validation.

    This implementation:
    - Loads memory-mapped data efficiently
    - Validates class distribution
    - Performs stratified splitting when possible
    - Returns None if data validation fails
    """

    def __init__(self, test_size: float = 0.30) -> None:
        """Initialize the splitter.

        Args:
            test_size: Fraction of data to use for testing.
        """
        self._test_size = test_size

    def split(
        self,
        X_mmap_path: str,
        y_mmap_path: str,
        seed: int,
        cv_folds: int,
    ) -> SplitData | None:
        """Split data into train and test sets with validation.

        Args:
            X_mmap_path: Path to memory-mapped feature data.
            y_mmap_path: Path to memory-mapped label data.
            seed: Random seed for reproducibility.
            cv_folds: Number of CV folds (used for validation).

        Returns:
            SplitData if successful, None if validation fails.
        """
        # Load memory-mapped data
        X_mmap = joblib.load(X_mmap_path, mmap_mode="r")
        y_mmap = joblib.load(y_mmap_path, mmap_mode="r")

        # Validate: Check minimum class counts
        _, counts = np.unique(y_mmap, return_counts=True)
        if counts.min() < 2:
            return None

        # Determine stratification
        stratify_y = y_mmap if counts.min() >= cv_folds else None

        # Split indices
        indices = np.arange(X_mmap.shape[0])
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self._test_size,
            stratify=stratify_y,
            random_state=seed,
        )

        # Validate training split sufficiency
        y_train_preview = y_mmap[train_idx]
        if len(np.unique(y_train_preview)) < 2:
            return None

        _, train_counts = np.unique(y_train_preview, return_counts=True)
        if train_counts.min() < cv_folds:
            return None

        # Materialize data
        return SplitData(
            X_train=X_mmap[train_idx],
            y_train=y_mmap[train_idx],
            X_test=X_mmap[test_idx],
            y_test=y_mmap[test_idx],
        )


__all__ = ["StratifiedDataSplitter"]
