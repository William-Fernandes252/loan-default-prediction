import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
)


def g_mean_score(y_true, y_pred):
    """Calculates the Geometric Mean of Sensitivity and Specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


g_mean_scorer = make_scorer(g_mean_score)
