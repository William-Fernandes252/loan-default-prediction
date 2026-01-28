"""Analysis module for experimental results.

This module provides types and protocols for evaluating and analyzing
experimental results from classification model training.
"""

from experiments.core.analysis.evaluation import (
    EvaluationMetrics,
    ModelResultsEvaluator,
)
from experiments.core.analysis.metrics import (
    IMBALANCE_RATIOS,
    Metric,
)
from experiments.core.analysis.translator import (
    Locale,
    Translator,
)

__all__ = [
    # Types
    "EvaluationMetrics",
    # Protocols
    "ModelResultsEvaluator",
    "Translator",
    # Metrics
    "Metric",
    "IMBALANCE_RATIOS",
    # i18n
    "Locale",
]
