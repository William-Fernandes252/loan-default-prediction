from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaggingClassifier


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
        cost_matrix: Optional[Dict[int, Any]] = None,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
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

        if self.cost_matrix is None:
            self.final_estimator_ = clone(self.base_estimator).fit(X, y)
            return self

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
            # Fallback if OOB score fails (e.g., small sample size)
            if np.any(np.sum(probas, axis=1) == 0):
                probas = bagging.predict_proba(X)
        else:
            probas = bagging.predict_proba(X)

        # Relabel based on expected cost
        # Cost matrix structure: {actual_class: cost_of_error}
        # Assuming binary classification: 0 and 1
        C_FP = self.cost_matrix.get(0, 1)  # Cost of False Positive (predicting 1 when 0)
        C_FN = self.cost_matrix.get(1, 1)  # Cost of False Negative (predicting 0 when 1)

        # Risk of predicting 0: P(1|x) * C_FN
        risk_0 = probas[:, 1] * C_FN
        # Risk of predicting 1: P(0|x) * C_FP
        risk_1 = probas[:, 0] * C_FP

        # Choose class with lower risk
        y_new_np = np.where(risk_1 < risk_0, 1, 0)

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
        return self.final_estimator_.predict_proba(X)
