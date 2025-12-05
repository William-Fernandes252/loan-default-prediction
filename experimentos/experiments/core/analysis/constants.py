"""Centralized constants for analysis module.

This module contains shared constants including metric configurations
and dataset-specific values used across the analysis pipeline.
"""

from dataclasses import dataclass

from experiments.core.analysis.protocols import TranslationFunc
from experiments.core.data import Dataset


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Metric configuration.

    A pure data container for metric metadata including the DataFrame column ID,
    translation key, and formatting specification.

    Attributes:
        id: The column name in the DataFrame (e.g., 'accuracy_balanced').
        display_name_key: The key for translation lookup.
        format_str: Format string for displaying values (default: "{:.4f}").
    """

    id: str
    display_name_key: str
    format_str: str = "{:.4f}"


def translate_metric(metric: MetricConfig, translate: TranslationFunc) -> str:
    """Translate a metric's display name.

    Args:
        metric: The metric configuration to translate.
        translate: Translation function for display names.

    Returns:
        The translated display name for the metric.
    """
    return translate(metric.display_name_key)


# Metric definitions - centralized to avoid duplication
METRIC_ACCURACY_BALANCED = MetricConfig(
    id="accuracy_balanced",
    display_name_key="Balanced Accuracy",
)

METRIC_G_MEAN = MetricConfig(
    id="g_mean",
    display_name_key="G-Mean",
)

METRIC_F1_SCORE = MetricConfig(
    id="f1_score",
    display_name_key="F1 Score",
)

METRIC_PRECISION = MetricConfig(
    id="precision",
    display_name_key="Precision",
)

METRIC_RECALL = MetricConfig(
    id="recall",
    display_name_key="Sensitivity",
)

METRIC_ROC_AUC = MetricConfig(
    id="roc_auc",
    display_name_key="ROC AUC",
)


# All available metrics
ALL_METRICS = [
    METRIC_ACCURACY_BALANCED,
    METRIC_G_MEAN,
    METRIC_F1_SCORE,
    METRIC_PRECISION,
    METRIC_RECALL,
    METRIC_ROC_AUC,
]


# Commonly used metric sets
STABILITY_METRICS = [
    METRIC_ACCURACY_BALANCED,
    METRIC_G_MEAN,
    METRIC_F1_SCORE,
    METRIC_RECALL,
]

SUMMARY_METRICS = [
    METRIC_ACCURACY_BALANCED,
    METRIC_G_MEAN,
    METRIC_F1_SCORE,
    METRIC_PRECISION,
    METRIC_RECALL,
]

IMBALANCE_METRICS = [
    METRIC_ACCURACY_BALANCED,
    METRIC_F1_SCORE,
    METRIC_G_MEAN,
    METRIC_RECALL,
]


def get_metric_configs(metric_ids: list[str] | None = None) -> list[MetricConfig]:
    """Get metric configurations by their IDs.

    Args:
        metric_ids: List of metric IDs to retrieve. If None, returns all metrics.

    Returns:
        List of MetricConfig objects for the requested metrics.
    """
    if metric_ids is None:
        return ALL_METRICS.copy()

    id_to_config = {m.id: m for m in ALL_METRICS}
    return [id_to_config[mid] for mid in metric_ids if mid in id_to_config]


def get_metric_display_names(
    translate: TranslationFunc,
    metric_ids: list[str] | None = None,
) -> dict[str, str]:
    """Get a mapping of metric IDs to their translated display names.

    Args:
        translate: Translation function.
        metric_ids: List of metric IDs to include. If None, includes all.

    Returns:
        Dictionary mapping metric ID -> translated display name.
    """
    configs = get_metric_configs(metric_ids)
    return {m.id: translate_metric(m, translate) for m in configs}


# Estimated Majority/Minority ratios for the datasets
# Used to populate the 'imbalance_ratio' column for cross-dataset analysis
IMBALANCE_RATIOS: dict[str, float] = {
    Dataset.LENDING_CLUB.id: 9.0,  # ~90% vs 10%
    Dataset.TAIWAN_CREDIT.id: 3.5,  # ~78% vs 22%
    Dataset.CORPORATE_CREDIT_RATING.id: 2000.0,  # ~99.95% vs 0.05%
}


# Default plot settings
DEFAULT_FIGURE_DPI = 300
DEFAULT_THEME_STYLE = "whitegrid"
