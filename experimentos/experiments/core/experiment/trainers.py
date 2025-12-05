"""Model training implementations for the experiment pipeline."""

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from experiments.core.experiment.protocols import SplitData, TrainedModel
from experiments.core.modeling.factories import (
    build_pipeline,
    get_hyperparameters,
    get_params_for_technique,
)
from experiments.core.modeling.types import ModelType, Technique


class GridSearchTrainer:
    """Trains models using GridSearchCV for hyperparameter optimization.

    This implementation:
    - Builds the appropriate pipeline for model type and technique
    - Performs grid search with cross-validation
    - Adjusts CV folds based on class distribution
    """

    def __init__(
        self,
        scoring: str = "roc_auc",
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize the trainer.

        Args:
            scoring: Scoring metric for optimization.
            n_jobs: Number of parallel jobs for grid search.
            verbose: Verbosity level.
        """
        self._scoring = scoring
        self._n_jobs = n_jobs
        self._verbose = verbose

    def train(
        self,
        data: SplitData,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        cv_folds: int,
        cost_grids: list[Any],
    ) -> TrainedModel:
        """Train and optimize a model using grid search.

        Args:
            data: The split training/test data.
            model_type: Type of model to train.
            technique: Technique for handling class imbalance.
            seed: Random seed for reproducibility.
            cv_folds: Number of cross-validation folds.
            cost_grids: Cost grid configurations.

        Returns:
            The trained model with best parameters.
        """
        # Build pipeline and parameter grid
        pipeline = build_pipeline(model_type, technique, seed)
        base_grid = get_hyperparameters(model_type)
        param_grid = get_params_for_technique(model_type, technique, base_grid, cost_grids)

        # Adjust CV if class count is low
        _, counts = np.unique(data.y_train, return_counts=True)
        actual_folds = max(2, min(cv_folds, counts.min()))

        # Configure grid search
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self._scoring,
            cv=StratifiedKFold(
                n_splits=actual_folds,
                shuffle=True,
                random_state=seed,
            ),
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        )

        # Fit and return
        grid.fit(data.X_train, data.y_train)

        return TrainedModel(
            estimator=grid.best_estimator_,
            best_params=grid.best_params_,
        )


__all__ = ["GridSearchTrainer"]
