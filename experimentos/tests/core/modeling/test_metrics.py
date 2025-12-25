import numpy as np

from experiments.core.modeling.metrics import g_mean_score


def test_g_mean_score_basic() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    score = g_mean_score(y_true, y_pred)
    assert np.isclose(score, np.sqrt(0.5))


def test_g_mean_score_handles_zero_division() -> None:
    y_true = [0, 0]
    y_pred = [0, 0]

    score = g_mean_score(y_true, y_pred)
    assert score == 0.0
