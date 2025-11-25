"""Custom estimators for cost-sensitive classification."""

from typing import Any, Optional, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


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


class RobustSVC(_ProbabilityMatrixClassesCorrectionMixin, SVC):
    """SVC that ensures output (N, 2) in predict_proba even in degenerate cases.

    Inherits from SVC, so it works natively with GridSearchCV.
    """

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return self._ensure_two_classes(probas, self.classes_)


class RobustXGBClassifier(_ProbabilityMatrixClassesCorrectionMixin, XGBClassifier):
    """XGBClassifier that ensures output (N, 2) in predict_proba even in degenerate cases.

    Inherits from XGBClassifier, so it works natively with GridSearchCV.
    """

    def predict_proba(self, X, **kwargs):
        probas = super().predict_proba(X, **kwargs)
        return self._ensure_two_classes(probas, self.classes_)


class MetaCostClassifier(ClassifierMixin, BaseEstimator):
    """
    A meta-classifier that makes a base classifier cost-sensitive.

    It uses bagging to estimate class probabilities and then relabels the training
    data to minimize expected cost.
    """

    _estimator_type = "classifier"
    final_estimator_: Optional[BaseEstimator]

    def __init__(
        self,
        base_estimator: BaseEstimator,
        cost_matrix: dict[int, Any] | str | None = None,
        n_estimators: int = 50,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.cost_matrix = cost_matrix
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.final_estimator_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("MetaCostClassifier only supports binary classification.")

        # 1. Handle "balanced" or None explicitly
        if self.cost_matrix is None:
            self.final_estimator_ = clone(self.base_estimator).fit(X, y)
            return self

        # Calculate costs dynamically if "balanced" is requested
        if self.cost_matrix == "balanced":
            # Compute weights: class 0 gets weight w0, class 1 gets weight w1
            weights = compute_class_weight(class_weight="balanced", classes=self.classes_, y=y)
            # Normalize so C_FP (cost of error on class 0) = 1.0
            # C_FN (cost of error on class 1) = w1 / w0
            w0, w1 = weights[0], weights[1]
            C_FP = 1.0
            C_FN = w1 / w0
        else:
            # Assume dict {class_label: cost_of_misclassifying_this_class}
            self.cost_matrix = cast(dict[int, Any], self.cost_matrix)
            C_FP = self.cost_matrix.get(0, 1)
            C_FN = self.cost_matrix.get(1, 1)

        # 2. Bagging for probability estimation
        if not hasattr(self.base_estimator, "predict_proba") and not hasattr(
            self.base_estimator, "decision_function"
        ):
            raise TypeError("Base estimator must support predict_proba or decision_function.")

        # Use Bagging to estimate probabilities
        bagging = BaggingClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=1,
        )
        bagging.fit(X, y)

        # Get probability estimates
        if (
            hasattr(bagging, "oob_decision_function_")
            and bagging.oob_decision_function_ is not None
        ):
            probas = bagging.oob_decision_function_
            # Fallback if OOB score fails (NaNs or zeros)
            if np.any(np.isnan(probas)) or np.any(np.sum(probas, axis=1) == 0):
                probas = bagging.predict_proba(X)
        else:
            probas = bagging.predict_proba(X)

        # 3. Relabeling logic
        # Risk of predicting 0: P(1|x) * Cost(FN)
        risk_0 = probas[:, 1] * C_FN
        # Risk of predicting 1: P(0|x) * Cost(FP)
        risk_1 = probas[:, 0] * C_FP

        # Choose class with lower risk
        y_new_np = np.where(risk_1 < risk_0, 1, 0)

        # 4. SAFETY CHECK: Prevent AdaBoost Crash
        # If relabeling makes the dataset single-class (e.g., all 1s), AdaBoost crashes.
        if len(np.unique(y_new_np)) < 2:
            # Fallback: Train a DummyClassifier to handle this gracefully
            # This effectively predicts the single class for everything.
            self.final_estimator_ = DummyClassifier(strategy="constant", constant=y_new_np[0])
            self.final_estimator_.fit(X, y_new_np)
        else:
            self.final_estimator_ = clone(self.base_estimator).fit(X, y_new_np)

        self.classes_ = self.final_estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.final_estimator_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.final_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.final_estimator_ is None:
            raise ValueError("The model has not been fitted yet.")

        probas = self.final_estimator_.predict_proba(X)

        # If the classifier collapsed to a single class (e.g., DummyClassifier),
        # the output will have shape (N, 1). We need to expand it to (N, 2).
        if probas.shape[1] == 1:
            n_samples = probas.shape[0]
            # Create a (N, 2) matrix filled with zeros
            new_probas = np.zeros((n_samples, 2), dtype=probas.dtype)

            # Discover which class remains (0 or 1)
            present_class = int(self.final_estimator_.classes_[0])

            # Map the probability to the correct column
            # If the class is 0, fill column 0; if it is 1, fill column 1.
            # It is assumed that the original classes were 0 and 1 (validated in fit).
            if present_class == 0:
                new_probas[:, 0] = probas[:, 0]  # P(0)
                # P(1) remains 0.0
            elif present_class == 1:
                new_probas[:, 1] = probas[:, 0]  # P(1)
                # P(0) remains 0.0

            return new_probas

        return probas
