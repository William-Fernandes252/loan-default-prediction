"""Factory functions and pipeline builders for machine learning models and techniques."""

from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiments.core.modeling.estimators import HAS_CUML, MetaCostClassifier
from experiments.core.modeling.estimators import RobustSVC as SVC
from experiments.core.modeling.estimators import RobustXGBClassifier as XGBClassifier
from experiments.core.modeling.types import ModelType, Technique

if HAS_CUML:
    from cuml.ensemble import RandomForestClassifier as CuRF

    from experiments.core.modeling.estimators import CuSVC


class DefaultEstimatorFactory:
    """Default implementation of EstimatorFactory protocol.

    Provides methods to create pipelines and parameter grids for models
    with different techniques.
    """

    def __init__(self, use_gpu: bool = False):
        self._use_gpu = use_gpu

        if self._use_gpu and not HAS_CUML:
            raise ImportError("cuML is not available, cannot use GPU-based estimators.")

    def _get_model_instance(self, model_type: ModelType, random_state: int) -> BaseEstimator:
        """Internal helper to dispatch to CPU or GPU classes."""

        if model_type == ModelType.SVM:
            if self._use_gpu and HAS_CUML:
                return CuSVC(random_state=random_state, probability=True)
            return SVC(random_state=random_state, probability=True)

        elif model_type == ModelType.RANDOM_FOREST:
            if self._use_gpu and HAS_CUML:
                return CuRF(random_state=random_state)
            return RandomForestClassifier(random_state=random_state, n_jobs=1)

        elif model_type == ModelType.XGBOOST:
            return XGBClassifier(
                random_state=random_state,
                device="cuda" if self._use_gpu else "cpu",
                eval_metric="logloss",
                n_jobs=1,
                objective="binary:logistic",
                tree_method="hist",
            )

        elif model_type == ModelType.MLP:
            return MLPClassifier(random_state=random_state)

        raise ValueError(f"Unknown model type: {model_type}")

    def create_pipeline(
        self,
        model_type: ModelType,
        technique: Technique,
        seed: int,
    ) -> ImbPipeline:
        """Create a pipeline for the given model type and technique.

        Args:
            model_type: Type of model to create.
            technique: Technique for handling class imbalance.
            seed: Random seed for reproducibility.

        Returns:
            The configured pipeline.
        """
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

        # Add sampling technique if applicable
        if technique == Technique.RANDOM_UNDER_SAMPLING:
            steps.append(("sampler", RandomUnderSampler(random_state=seed)))
        elif technique == Technique.SMOTE:
            steps.append(("sampler", SMOTE(random_state=seed)))
        elif technique == Technique.SMOTE_TOMEK:
            steps.append(("sampler", SMOTETomek(random_state=seed)))
        # Add classifier
        clf = self._get_model_instance(model_type, seed)

        # Wrap with MetaCost if applicable
        if technique == Technique.META_COST:
            clf = MetaCostClassifier(base_estimator=clf, random_state=seed)

        steps.append(("clf", clf))
        return ImbPipeline(steps)

    def get_param_grid(
        self,
        model_type: ModelType,
        technique: Technique,
        cost_grids: list[Any],
    ) -> list[dict[str, Any]]:
        """Get parameter grid for the given model type and technique.

        Args:
            model_type: Type of model.
            technique: Technique for handling class imbalance.
            cost_grids: Cost grid configurations.

        Returns:
            Parameter grid for hyperparameter search.
        """
        base_grid = self._get_hyperparameters(model_type)
        return self._get_params_for_technique(model_type, technique, base_grid, cost_grids)

    def _get_hyperparameters(self, model_type: ModelType) -> dict:
        """Get hyperparameters for the given model type."""
        params: dict[ModelType, dict[str, Any]] = {
            ModelType.SVM: {
                "clf__C": [0.1, 1, 10, 100],
                "clf__kernel": ["rbf"],  # Linear can be too slow for large datasets
                "clf__probability": [True],  # Necessary for some metrics or MetaCost
                # For CS-SVM, the weight will be injected dynamically or via grid here
                # If it is Baseline, class_weight is None.
            },
            ModelType.RANDOM_FOREST: {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 5],
                # Random Forest also accepts class_weight, useful for comparison with CS-SVM
            },
            ModelType.XGBOOST: {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.1, 0.3],  # 'eta' in XGBoost terminology
                "clf__max_depth": [3, 6, 10],  # Important for controlling complexity
                "clf__subsample": [0.8, 1.0],  # Helps to avoid over-fitting
                "clf__colsample_bytree": [0.8, 1.0],  # Fraction of features per tree
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
                if key.startswith("clf__"):
                    # Transform: clf__param -> clf__base_estimator__param
                    new_key = key.replace("clf__", "clf__base_estimator__", 1)
                    meta_cost_params[new_key] = value
                else:
                    # Keep non-classifier params (e.g. sampler params) as is
                    meta_cost_params[key] = value

            return [meta_cost_params]

        return [new_params]
