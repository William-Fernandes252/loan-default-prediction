"""Tests for the analysis constants module."""

import pytest

from experiments.core.analysis.constants import (
    ALL_METRICS,
    IMBALANCE_METRICS,
    IMBALANCE_RATIOS,
    METRIC_ACCURACY_BALANCED,
    METRIC_F1_SCORE,
    METRIC_G_MEAN,
    METRIC_PRECISION,
    METRIC_RECALL,
    METRIC_ROC_AUC,
    STABILITY_METRICS,
    SUMMARY_METRICS,
    MetricConfig,
    get_metric_configs,
    get_metric_display_names,
)
from experiments.core.data import Dataset


def identity_translate(s: str) -> str:
    """Identity translation function for testing."""
    return s


class DescribeMetricConfig:
    def it_stores_id_and_display_name_key(self):
        config = MetricConfig(id="test_metric", display_name_key="Test Metric")

        assert config.id == "test_metric"
        assert config.display_name_key == "Test Metric"

    def it_has_default_format_string(self):
        config = MetricConfig(id="test", display_name_key="Test")

        assert config.format_str == "{:.4f}"

    def it_allows_custom_format_string(self):
        config = MetricConfig(id="test", display_name_key="Test", format_str="{:.2%}")

        assert config.format_str == "{:.2%}"

    def it_returns_translated_display_name(self):
        config = MetricConfig(id="test", display_name_key="Test Metric")

        def custom_translate(s: str) -> str:
            return f"[{s}]"

        assert config.get_display_name(custom_translate) == "[Test Metric]"

    def it_is_frozen_and_hashable(self):
        config = MetricConfig(id="test", display_name_key="Test")

        with pytest.raises(AttributeError):
            config.id = "changed"  # type: ignore

        # Should be hashable (can be used in sets/dicts)
        metric_set = {config}
        assert config in metric_set


class DescribePredefinedMetrics:
    def it_has_accuracy_balanced_metric(self):
        assert METRIC_ACCURACY_BALANCED.id == "accuracy_balanced"
        assert METRIC_ACCURACY_BALANCED.display_name_key == "Balanced Accuracy"

    def it_has_g_mean_metric(self):
        assert METRIC_G_MEAN.id == "g_mean"
        assert METRIC_G_MEAN.display_name_key == "G-Mean"

    def it_has_f1_score_metric(self):
        assert METRIC_F1_SCORE.id == "f1_score"
        assert METRIC_F1_SCORE.display_name_key == "F1 Score"

    def it_has_precision_metric(self):
        assert METRIC_PRECISION.id == "precision"
        assert METRIC_PRECISION.display_name_key == "Precision"

    def it_has_recall_metric(self):
        assert METRIC_RECALL.id == "recall"
        assert METRIC_RECALL.display_name_key == "Sensitivity"

    def it_has_roc_auc_metric(self):
        assert METRIC_ROC_AUC.id == "roc_auc"
        assert METRIC_ROC_AUC.display_name_key == "ROC AUC"

    def it_has_all_metrics_list(self):
        assert len(ALL_METRICS) == 6
        assert METRIC_ACCURACY_BALANCED in ALL_METRICS
        assert METRIC_G_MEAN in ALL_METRICS
        assert METRIC_F1_SCORE in ALL_METRICS
        assert METRIC_PRECISION in ALL_METRICS
        assert METRIC_RECALL in ALL_METRICS
        assert METRIC_ROC_AUC in ALL_METRICS


class DescribeMetricSets:
    def it_has_stability_metrics(self):
        assert len(STABILITY_METRICS) == 4
        assert METRIC_ACCURACY_BALANCED in STABILITY_METRICS
        assert METRIC_G_MEAN in STABILITY_METRICS
        assert METRIC_F1_SCORE in STABILITY_METRICS
        assert METRIC_RECALL in STABILITY_METRICS

    def it_has_summary_metrics(self):
        assert len(SUMMARY_METRICS) == 5
        assert METRIC_ACCURACY_BALANCED in SUMMARY_METRICS
        assert METRIC_G_MEAN in SUMMARY_METRICS
        assert METRIC_F1_SCORE in SUMMARY_METRICS
        assert METRIC_PRECISION in SUMMARY_METRICS
        assert METRIC_RECALL in SUMMARY_METRICS

    def it_has_imbalance_metrics(self):
        assert len(IMBALANCE_METRICS) == 4
        assert METRIC_ACCURACY_BALANCED in IMBALANCE_METRICS
        assert METRIC_F1_SCORE in IMBALANCE_METRICS
        assert METRIC_G_MEAN in IMBALANCE_METRICS
        assert METRIC_RECALL in IMBALANCE_METRICS


class DescribeGetMetricConfigs:
    def it_returns_all_metrics_when_no_ids_provided(self):
        result = get_metric_configs(None)

        assert len(result) == len(ALL_METRICS)
        assert result == ALL_METRICS

    def it_returns_copy_of_all_metrics(self):
        result = get_metric_configs(None)
        result.append(MetricConfig(id="new", display_name_key="New"))

        assert len(ALL_METRICS) == 6  # Original unchanged

    def it_returns_specific_metrics_by_id(self):
        result = get_metric_configs(["accuracy_balanced", "f1_score"])

        assert len(result) == 2
        assert result[0].id == "accuracy_balanced"
        assert result[1].id == "f1_score"

    def it_ignores_unknown_metric_ids(self):
        result = get_metric_configs(["accuracy_balanced", "unknown_metric"])

        assert len(result) == 1
        assert result[0].id == "accuracy_balanced"

    def it_returns_empty_list_for_all_unknown_ids(self):
        result = get_metric_configs(["unknown1", "unknown2"])

        assert result == []


class DescribeGetMetricDisplayNames:
    def it_returns_all_display_names_when_no_ids_provided(self):
        result = get_metric_display_names(identity_translate, None)

        assert len(result) == 6
        assert result["accuracy_balanced"] == "Balanced Accuracy"
        assert result["g_mean"] == "G-Mean"

    def it_returns_specific_display_names_by_id(self):
        result = get_metric_display_names(identity_translate, ["accuracy_balanced", "f1_score"])

        assert len(result) == 2
        assert result["accuracy_balanced"] == "Balanced Accuracy"
        assert result["f1_score"] == "F1 Score"

    def it_applies_translation_function(self):
        def upper_translate(s: str) -> str:
            return s.upper()

        result = get_metric_display_names(upper_translate, ["accuracy_balanced"])

        assert result["accuracy_balanced"] == "BALANCED ACCURACY"


class DescribeImbalanceRatios:
    def it_has_ratios_for_all_datasets(self):
        for dataset in Dataset:
            assert dataset.id in IMBALANCE_RATIOS

    def it_has_lending_club_ratio(self):
        assert IMBALANCE_RATIOS[Dataset.LENDING_CLUB.id] == 9.0

    def it_has_taiwan_credit_ratio(self):
        assert IMBALANCE_RATIOS[Dataset.TAIWAN_CREDIT.id] == 3.5

    def it_has_corporate_credit_rating_ratio(self):
        assert IMBALANCE_RATIOS[Dataset.CORPORATE_CREDIT_RATING.id] == 2000.0
