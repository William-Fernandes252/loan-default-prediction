from typing import Any, cast

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from experiments.core.modeling.estimators import (
    MetaCostClassifier,
    _ProbabilityMatrixClassesCorrectionMixin,
)


class NoProbaEstimator:
    """Minimal estimator lacking predict_proba and decision_function."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NoProbaEstimator":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def get_params(self, deep: bool = False) -> dict[str, Any]:  # scikit compatibility
        return {}

    def set_params(self, **params: Any) -> "NoProbaEstimator":  # scikit compatibility
        return self


def test_probability_matrix_correction_handles_single_class_zero() -> None:
    probas = np.array([[0.7], [0.2]])
    corrected = _ProbabilityMatrixClassesCorrectionMixin._ensure_two_classes(probas, np.array([0]))

    assert corrected.shape == (2, 2)
    assert np.allclose(corrected[:, 0], [0.7, 0.2])
    assert np.allclose(corrected[:, 1], [0.0, 0.0])


def test_probability_matrix_correction_handles_single_class_one() -> None:
    probas = np.array([[0.3], [0.9]])
    corrected = _ProbabilityMatrixClassesCorrectionMixin._ensure_two_classes(probas, np.array([1]))

    assert corrected.shape == (2, 2)
    assert np.allclose(corrected[:, 1], [0.3, 0.9])
    assert np.allclose(corrected[:, 0], [0.0, 0.0])


def test_meta_cost_classifier_basic_fit_and_predict() -> None:
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])

    clf = MetaCostClassifier(
        base_estimator=DummyClassifier(strategy="most_frequent"), cost_matrix=None
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape


def test_meta_cost_classifier_expands_single_class_predict_proba() -> None:
    X = np.array([[0], [0], [0]])
    y = np.array([0, 0, 0])

    clf = MetaCostClassifier(
        base_estimator=DummyClassifier(strategy="most_frequent"), cost_matrix=None
    )
    clf.fit(X, y)

    probas = clf.predict_proba(X)
    assert probas.shape == (3, 2)
    assert np.allclose(probas[:, 0], np.ones(3))
    assert np.allclose(probas[:, 1], np.zeros(3))


def test_meta_cost_classifier_rejects_multiclass() -> None:
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])

    clf = MetaCostClassifier(base_estimator=DummyClassifier(), cost_matrix=None)

    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_meta_cost_classifier_requires_proba_or_decision_function() -> None:
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    clf = MetaCostClassifier(base_estimator=cast(Any, NoProbaEstimator()), cost_matrix={})

    with pytest.raises(TypeError):
        clf.fit(X, y)


def test_meta_cost_classifier_balanced_cost_matrix_trains() -> None:
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])

    clf = MetaCostClassifier(
        base_estimator=DummyClassifier(strategy="most_frequent"),
        cost_matrix="balanced",
        n_estimators=5,
        random_state=0,
    )

    clf.fit(X, y)
    assert clf.final_estimator_ is not None
    assert hasattr(clf.final_estimator_, "predict")


def test_meta_cost_classifier_accepts_base_estimator_with_decision_function_only() -> None:
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    clf = MetaCostClassifier(base_estimator=SVC(probability=False), cost_matrix=None)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape
