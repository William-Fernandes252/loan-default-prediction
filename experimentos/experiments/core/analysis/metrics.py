"""Centralized constants for analysis module.

This module contains shared constants including metric configurations
and dataset-specific values used across the analysis pipeline.
"""

import enum

from experiments.core.data import Dataset

IMBALANCE_RATIOS: dict[Dataset, float] = {
    Dataset.LENDING_CLUB: 9.0,  # ~90% vs 10%
    Dataset.TAIWAN_CREDIT: 3.5,  # ~78% vs 22%
    Dataset.CORPORATE_CREDIT_RATING: 2000.0,  # ~99.95% vs 0.05%
}
"""Estimated Majority/Minority ratios for the datasets

Used to populate the `imbalance_ratio` column for cross-dataset analysis
"""


class Metric(enum.StrEnum):
    """Enumeration of metrics used to evaluate the models trained in the experiments."""

    ACCURACY_BALANCED = "accuracy_balanced"
    G_MEAN = "g_mean"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
