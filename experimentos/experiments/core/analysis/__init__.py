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

__all__ = [
    # Types
    "EvaluationMetrics",
    # Protocols
    "ModelResultsEvaluator",
    # Metrics
    "Metric",
    "IMBALANCE_RATIOS",
]
