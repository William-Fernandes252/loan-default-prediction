"""Implementation of model training using grid-search to optimize hyperparameters."""

from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from experiments.core.modeling.classifiers import Classifier, ModelType, Technique
from experiments.core.training.trainers import ModelTrainRequest, TrainedModel


class GridSearchModelTrainer:
    """Trains models using `GridSearchCV` for hyperparameter optimization.

    This implementation:
    - Uses an injected EstimatorFactory to build pipelines and param grids
    - Performs grid search with cross-validation
    - Adjusts CV folds based on class distribution
    """

    def __init__(
        self,
        cost_grids: list[Any],
        scoring: str = "balanced_accuracy",
        cv_folds: int = 5,
        verbose: bool = False,
    ) -> None:
        """Initialize the trainer.

        Args:
            cost_grids (list[Any]): Cost grid configurations.
            scoring (str): Scoring metric for optimization. Defaults to "balanced_accuracy".
            cv_folds (int): Number of cross-validation folds. Defaults to 5.
            n_jobs (int): Number of parallel jobs for grid search. Defaults to 1.
            verbose (bool): Verbosity flag. Defaults to False.
        """
        self._scoring = scoring
        self._verbose = verbose
        self._cost_grids = cost_grids
        self._cv_folds = cv_folds

    def train(
        self,
        request: ModelTrainRequest,
        n_jobs: int = 1,
    ) -> TrainedModel:
        """Train and optimize a model using grid search.

        Args:
            request (ModelTrainRequest): The training request data.
            n_jobs (int): Number of parallel jobs for grid search. Defaults to 1.

        Returns:
            TrainedModel: The trained model with best parameters.
        """
        # Adjust CV if class count is low
        _, counts = np.unique(request.data.y_train, return_counts=True)
        actual_folds = max(2, min(self._cv_folds, counts.min()))

        # Configure grid search
        grid = GridSearchCV(
            estimator=cast(BaseEstimator, request.classifier),
            param_grid=self.get_params_for_classifier(
                request.model_type,
                request.technique,
                self._cost_grids,
            ),
            scoring=self._scoring,
            cv=StratifiedKFold(
                n_splits=actual_folds,
                shuffle=True,
                random_state=request.seed,
            ),
            n_jobs=n_jobs,
            verbose=self._verbose,
            error_score="raise",
            return_train_score=False,
        )

        # Fit and return
        grid.fit(request.data.X_train, request.data.y_train)

        return TrainedModel(
            model=cast(Classifier, grid.best_estimator_),
            params=grid.best_params_,
            seed=request.seed,
        )

    def get_params_for_classifier(
        self,
        model_type: ModelType,
        technique: Technique,
        cost_grids: list[Any],
    ) -> list[dict[str, Any]]:
        """Get the best parameters for a given classifier.

        Args:
            model_type (ModelType): The type of the model.
            technique (Technique): The technique used by the model.
            cost_grids (list[Any]): Cost grid configurations.

        Returns:
            list[dict[str, Any]]: The best hyperparameters found.
        """
        base_grid = self._get_hyperparameters(model_type)
        return self._get_params_for_technique(model_type, technique, base_grid, cost_grids)

    def _get_hyperparameters(self, model_type: ModelType) -> dict:
        """Get hyperparameters for the given model type."""
        params: dict[ModelType, dict[str, Any]] = {
            ModelType.SVM: {
                "clf__loss": ["log_loss"],
                "clf__alpha": [1e-4, 1e-3, 1e-2],
                "clf__penalty": ["l2", "elasticnet"],
                "clf__max_iter": [1000, 2000],
                "clf__tol": [1e-3],
            },
            ModelType.RANDOM_FOREST: {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10, 20],
                "clf__min_samples_leaf": [1, 5],
                # Random Forest also accepts class_weight, useful for comparison with CS-SVM
            },
            ModelType.XGBOOST: {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.1],  # Removed 0.3 to save time
                "clf__max_depth": [3, 6, 10],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_alpha": [0, 0.1],  # Test: None vs. Sparse (L1)
                "clf__reg_lambda": [1.0, 10.0],  # Test: Default vs. Strong (L2)
            },
            ModelType.MLP: {
                "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "clf__activation": ["relu", "tanh"],
                "clf__alpha": [0.0001, 0.01],
                "clf__early_stopping": [True],
            },
        }
        return params.get(model_type, {})

    @staticmethod
    def _get_params_for_technique(
        model_type: ModelType,
        technique: Technique,
        base_params: dict,
        cost_matrix: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Adjusts the parameter grid based on the technique."""
        new_params: dict[str, Any] = base_params.copy()

        # 1. Handle CS-SVM (Cost-Sensitive SVM)
        if technique == Technique.CS_SVM or (
            technique == Technique.BASELINE
            and model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]
            and technique != Technique.META_COST
        ):
            if technique == Technique.CS_SVM and model_type == ModelType.SVM:
                new_params["clf__class_weight"] = cost_matrix

        # 2. Handle MetaCost
        if technique == Technique.META_COST:
            # Start with the cost matrix (parameter of MetaCost itself)
            meta_cost_params = {"clf__cost_matrix": cost_matrix}

            # Iterate through the base parameters (e.g., learning_rate, hidden_layer_sizes)
            # and "push" them down into the base_estimator
            for key, value in base_params.items():
                if isinstance(key, str) and key.startswith("clf__"):
                    # Transform: clf__param -> clf__base_estimator__param
                    new_key = key.replace("clf__", "clf__base_estimator__", 1)
                    meta_cost_params[new_key] = value
                else:
                    # Keep non-classifier params (e.g. sampler params) as is
                    meta_cost_params[key] = value

            return [meta_cost_params]

        return [new_params]


__all__ = ["GridSearchModelTrainer"]
