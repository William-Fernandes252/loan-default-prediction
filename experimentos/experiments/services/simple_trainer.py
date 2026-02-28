"""Implementation of model training using fixed default hyperparameters.

This trainer is designed for rapid prototyping and testing, using sensible
default hyperparameters without grid search optimization.
"""

from typing import Any

from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.training.trainers import ModelTrainRequest, TrainedModel


class SimpleModelTrainer:
    """Trains models with fixed default hyperparameters (no grid search).

    This trainer is designed for rapid prototyping and testing. It uses
    sensible default hyperparameters for each model type, avoiding the
    computational overhead of grid search cross-validation.

    Features:
    - Single fixed hyperparameter configuration per model type
    - No cross-validation (direct fit on training data)
    - Handles technique-specific adjustments (CS-SVM, MLP + RUS)
    - 5-10x faster than GridSearchModelTrainer

    Trade-offs:
    - No hyperparameter optimization
    - May achieve suboptimal performance compared to tuned models
    - Best for quick experiments and debugging
    """

    def __init__(self, cost_grids: list[Any] | None = None) -> None:
        """Initialize the simple trainer.

        Args:
            cost_grids (list[Any] | None): Cost grid configurations for CS-SVM.
                Defaults to None.
        """
        self._cost_grids = cost_grids

    def train(
        self,
        request: ModelTrainRequest,
        n_jobs: int = 1,
    ) -> TrainedModel:
        """Train a model using default hyperparameters.

        Args:
            request (ModelTrainRequest): The training request data.
            n_jobs (int): Number of parallel jobs (ignored for compatibility).

        Returns:
            TrainedModel: The trained model with default parameters.
        """
        # Get default params
        params = self._get_default_params(request.model_type)

        # Apply technique adjustments
        params = self._apply_technique_adjustments(params, request.model_type, request.technique)

        # Set parameters on classifier
        request.classifier.set_params(**params)

        # Fit directly on training data (no GridSearchCV)
        request.classifier.fit(request.data.X_train, request.data.y_train)

        # Return trained model
        return TrainedModel(
            model=request.classifier,
            params=params,
            seed=request.seed,
        )

    def _get_default_params(self, model_type: ModelType) -> dict[str, Any]:
        """Get default hyperparameters for the given model type.

        Args:
            model_type (ModelType): The type of the model.

        Returns:
            dict[str, Any]: Default hyperparameters for the model type.
        """
        params: dict[ModelType, dict[str, Any]] = {
            ModelType.SVM: {
                "clf__loss": "log_loss",
                "clf__alpha": 0.0001,
                "clf__penalty": "l2",
                "clf__max_iter": 1000,
                "clf__tol": 1e-3,
            },
            ModelType.RANDOM_FOREST: {
                "clf__n_estimators": 100,
                "clf__max_depth": None,
                "clf__min_samples_leaf": 1,
            },
            ModelType.XGBOOST: {
                "clf__n_estimators": 100,
                "clf__learning_rate": 0.1,
                "clf__max_depth": 6,
                "clf__subsample": 1.0,
                "clf__colsample_bytree": 1.0,
                "clf__reg_alpha": 0,
                "clf__reg_lambda": 1.0,
            },
            ModelType.MLP: {
                "clf__hidden_layer_sizes": (100,),
                "clf__activation": "relu",
                "clf__alpha": 0.0001,
                "clf__early_stopping": True,
            },
        }
        return params.get(model_type, {})

    def _apply_technique_adjustments(
        self,
        params: dict[str, Any],
        model_type: ModelType,
        technique: Technique,
    ) -> dict[str, Any]:
        """Apply technique-specific parameter adjustments.

        Handles:
        - CS-SVM: Add class_weight from first cost grid entry
        - MLP + RandomUnderSampling: Disable early_stopping

        Args:
            params (dict[str, Any]): Base hyperparameters.
            model_type (ModelType): The type of the model.
            technique (Technique): The technique used by the model.

        Returns:
            dict[str, Any]: Adjusted hyperparameters.
        """
        new_params: dict[str, Any] = params.copy()

        # Handle CS-SVM (Cost-Sensitive SVM)
        if technique == Technique.CS_SVM and model_type == ModelType.SVM:
            new_params["clf__class_weight"] = self._resolve_cost_weight()

        # Handle MLP + RandomUnderSampling
        # After RUS inside CV folds, fold-train sets can become very small.
        # MLP with early_stopping=True creates an internal stratified split,
        # which can fail with: "test_size = 1 should be >= number of classes = 2".
        if technique == Technique.RANDOM_UNDER_SAMPLING and model_type == ModelType.MLP:
            new_params["clf__early_stopping"] = False

        return new_params

    def _resolve_cost_weight(self) -> Any:
        """Resolve the cost weight for CS-SVM technique.

        Picks the first non-None cost grid entry, defaulting to ``"balanced"``
        when no usable entry exists.  ``None`` entries (meaning *no* class
        weighting) are skipped because they would defeat the purpose of the
        CS-SVM technique.

        Returns:
            The resolved cost weight value.
        """
        if not self._cost_grids:
            return "balanced"
        return next((cg for cg in self._cost_grids if cg is not None), "balanced")


__all__ = ["SimpleModelTrainer"]
