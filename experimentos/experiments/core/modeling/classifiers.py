"""Custom estimators for cost-sensitive classification."""

import enum
from typing import Any, Protocol, override

import numpy as np
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as CuRF
    from cuml.svm import SVC as CuSVC

    cuml.set_global_output_type("numpy")

    HAS_CUML = True
except ImportError:
    HAS_CUML = False


class _ProbabilityMatrixClassesCorrectionMixin:
    @staticmethod
    def _ensure_two_classes(probas: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Ensures that the probability matrix has 2 columns, even if the model
        has collapsed to a single class.
        """
        if probas.shape[1] == 1:
            n_samples = probas.shape[0]
            new_probas = np.zeros((n_samples, 2), dtype=probas.dtype)

            # The model only knows one class. Which one is it?
            present_class = int(classes[0])

            # If the present class is 0, fill column 0. If it is 1, fill column 1.
            # Assumes binary classes {0, 1}.
            if present_class == 0:
                new_probas[:, 0] = probas[:, 0]
                new_probas[:, 1] = 0.0
            elif present_class == 1:
                new_probas[:, 1] = probas[:, 0]
                new_probas[:, 0] = 0.0

            return new_probas
        return probas


class RobustSVC(_ProbabilityMatrixClassesCorrectionMixin, SGDClassifier):
    """SVC that ensures output (N, 2) in predict_proba even in degenerate cases.

    Inherits from SVC, so it works natively with GridSearchCV.
    """

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return self._ensure_two_classes(probas, self.classes_)


if HAS_CUML:

    class RobustCuSVC(_ProbabilityMatrixClassesCorrectionMixin, CuSVC):
        """GPU-accelerated Robust SVC using RAPIDS cuML."""

        def predict_proba(self, X):
            probas = super().predict_proba(X)

            # Convert to numpy for the Mixin logic (which uses np.zeros/np.where)
            # This ensures compatibility with the rest of your scikit-learn pipeline
            if hasattr(probas, "get"):
                probas = probas.get()

            return self._ensure_two_classes(probas, self.classes_)  # type: ignore[arg-type]


class RobustXGBClassifier(_ProbabilityMatrixClassesCorrectionMixin, XGBClassifier):
    """XGBClassifier that ensures output (N, 2) in predict_proba even in degenerate cases.

    Inherits from XGBClassifier, so it works natively with GridSearchCV.
    """

    @override
    def fit(self, X, y, **kwargs):
        if getattr(self, "base_score", None) is None:
            self.set_params(base_score=0.5)
        return super().fit(X, y, **kwargs)

    def predict_proba(self, X, **kwargs):  # type: ignore[override]
        probas = super().predict_proba(X, **kwargs)
        return self._ensure_two_classes(probas, self.classes_)


class ModelType(enum.StrEnum):
    """Types of models used in experiments."""

    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    XGBOOST = "xgboost"
    MLP = "mlp"


class Technique(enum.StrEnum):
    """Types of techniques used in experiments."""

    BASELINE = "baseline"
    SMOTE = "smote"
    RANDOM_UNDER_SAMPLING = "random_under_sampling"
    SMOTE_TOMEK = "smote_tomek"
    CS_SVM = "cs_svm"


class Classifier(Protocol):
    """Protocol for classifier models, based on scikit-learn."""

    def fit(self, X: Any, y: Any) -> None: ...

    def predict(self, X: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


class ClassifierFactory(Protocol):
    """Creates classifiers from model type and technique."""

    def create_model(
        self,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        use_gpu: bool | None = None,
        n_jobs: int = 1,
    ) -> Classifier:
        """Create a classifier for the given model type and technique.

        Args:
            model_type: Type of model to create.
            technique: Technique for handling class imbalance.
            seed: Random seed for reproducibility.
            use_gpu: Whether to use GPU for training. Defaults to `None`, which means using the environment setting.
            n_jobs: Number of parallel jobs for training. Defaults to `1`.

        Returns:
            The configured classifier.

        Raises:
            ValueError: If the model type is unknown.
        """
        ...


__all__ = [
    "CuRF",
    "RobustSVC",
    "RobustXGBClassifier",
    "ModelType",
    "Technique",
    "Classifier",
    "ClassifierFactory",
    "HAS_CUML",
]
