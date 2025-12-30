"""Tests for experiments.core.experiment.evaluators module."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from experiments.core.experiment.evaluators import ClassificationEvaluator
from experiments.core.experiment.protocols import EvaluationResult


@pytest.fixture
def fitted_classifier() -> LogisticRegression:
    """Create a fitted logistic regression classifier."""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.array([0] * 50 + [1] * 50)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    return clf


@pytest.fixture
def test_data() -> tuple[np.ndarray, np.ndarray]:
    """Create test data for evaluation."""
    np.random.seed(42)
    X_test = np.random.rand(30, 10)
    y_test = np.array([0] * 15 + [1] * 15)
    return X_test, y_test


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Create test data with perfect predictions."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return y_true, y_pred


class DescribeClassificationEvaluator:
    """Tests for ClassificationEvaluator class."""

    def it_returns_evaluation_result(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify evaluate returns an EvaluationResult."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert isinstance(result, EvaluationResult)
        assert isinstance(result.metrics, dict)

    def it_computes_balanced_accuracy(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify balanced accuracy is computed."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert "accuracy_balanced" in result.metrics
        assert 0.0 <= result.metrics["accuracy_balanced"] <= 1.0  # type: ignore[operator]

    def it_computes_g_mean(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify G-mean is computed."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert "g_mean" in result.metrics
        assert 0.0 <= result.metrics["g_mean"] <= 1.0  # type: ignore[operator]

    def it_computes_f1_score(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify F1 score is computed."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert "f1_score" in result.metrics
        assert 0.0 <= result.metrics["f1_score"] <= 1.0  # type: ignore[operator]

    def it_computes_precision(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify precision is computed."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert "precision" in result.metrics
        assert 0.0 <= result.metrics["precision"] <= 1.0  # type: ignore[operator]

    def it_computes_recall(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify recall is computed."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert "recall" in result.metrics
        assert 0.0 <= result.metrics["recall"] <= 1.0  # type: ignore[operator]

    def it_computes_roc_auc(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify ROC AUC is computed."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        assert "roc_auc" in result.metrics
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0  # type: ignore[operator]

    def it_returns_all_six_metrics(
        self,
        fitted_classifier: LogisticRegression,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify all six metrics are returned."""
        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(fitted_classifier, X_test, y_test)

        expected_metrics = [
            "accuracy_balanced",
            "g_mean",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
        ]
        assert all(m in result.metrics for m in expected_metrics)
        assert len(result.metrics) == 6


class DescribeClassificationEvaluatorEdgeCases:
    """Tests for ClassificationEvaluator edge cases."""

    def it_handles_model_without_predict_proba(
        self,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify evaluator handles models without predict_proba."""
        # SVC with default kernel doesn't have predict_proba
        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.array([0] * 50 + [1] * 50)

        clf = SVC(kernel="linear", random_state=42)
        clf.fit(X_train, y_train)

        X_test, y_test = test_data
        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(clf, X_test, y_test)

        # Should default to 0.5 for AUC when predict_proba is not available
        assert result.metrics["roc_auc"] == 0.5

    def it_handles_perfect_predictions(self) -> None:
        """Verify evaluator handles perfect predictions."""
        np.random.seed(42)
        X_train = np.vstack([np.zeros((50, 10)), np.ones((50, 10))])
        y_train = np.array([0] * 50 + [1] * 50)
        X_test = np.vstack([np.zeros((10, 10)), np.ones((10, 10))])
        y_test = np.array([0] * 10 + [1] * 10)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)

        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(clf, X_test, y_test)

        # Perfect predictions should give high scores
        assert result.metrics["accuracy_balanced"] >= 0.9  # type: ignore[operator]
        assert result.metrics["f1_score"] >= 0.9  # type: ignore[operator]

    def it_handles_all_same_predictions(self) -> None:
        """Verify evaluator handles when model predicts same class for all."""
        # Train a real model on separable data, but test on mixed data
        np.random.seed(42)
        X_train = np.vstack([np.zeros((50, 10)), np.ones((50, 10))])
        y_train = np.array([0] * 50 + [1] * 50)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)

        # Test on all zeros - model will predict all 0s
        X_test = np.zeros((20, 10))
        y_test = np.array([0] * 10 + [1] * 10)

        evaluator = ClassificationEvaluator()

        result = evaluator.evaluate(clf, X_test, y_test)

        # Should still compute metrics without errors
        assert "accuracy_balanced" in result.metrics
        assert "precision" in result.metrics
        # With all predictions being 0, recall for class 1 is 0
        assert result.metrics["recall"] == 0.0

    def it_uses_probability_predictions_for_auc(
        self,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify ROC AUC uses probability predictions with a well-calibrated model."""
        X_test, y_test = test_data

        # Train a model on highly separable data to get good probabilities
        np.random.seed(42)
        X_train = np.vstack(
            [
                np.random.rand(50, 10) * 0.3,  # Class 0: small values
                np.random.rand(50, 10) * 0.7 + 0.3,  # Class 1: larger values
            ]
        )
        y_train = np.array([0] * 50 + [1] * 50)

        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)

        evaluator = ClassificationEvaluator()
        result = evaluator.evaluate(clf, X_test, y_test)

        # With a real model, AUC should be computed from probabilities
        # and should be non-trivial (not 0.5 which indicates no discrimination)
        assert "roc_auc" in result.metrics
        assert result.metrics["roc_auc"] != 0.5
