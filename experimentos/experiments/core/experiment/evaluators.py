"""Model evaluation implementations for the experiment pipeline."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.core.experiment.protocols import EvaluationResult
from experiments.core.modeling.metrics import g_mean_score


class ClassificationEvaluator:
    """Evaluates classification models using multiple metrics.

    This implementation computes:
    - Balanced accuracy
    - G-mean
    - F1 score
    - Precision
    - Recall
    - ROC AUC
    """

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> EvaluationResult:
        """Evaluate a trained model on test data.

        Args:
            model: The trained model.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Evaluation result with all computed metrics.
        """
        y_pred = model.predict(X_test)

        # Try to get probability predictions for AUC
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        except (AttributeError, IndexError):
            auc_score = 0.5

        metrics = {
            "accuracy_balanced": balanced_accuracy_score(y_test, y_pred),
            "g_mean": g_mean_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": auc_score,
        }

        return EvaluationResult(metrics=metrics)


__all__ = ["ClassificationEvaluator"]
