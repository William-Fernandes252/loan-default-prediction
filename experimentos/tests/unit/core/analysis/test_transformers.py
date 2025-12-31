"""Tests for the analysis transformers module."""

import pandas as pd
import pytest

from experiments.core.analysis.constants import IMBALANCE_RATIOS
from experiments.core.analysis.transformers import (
    CostSensitiveVsResamplingTransformer,
    ExperimentSummaryTransformer,
    HyperparameterTransformer,
    ImbalanceImpactTransformer,
    RiskTradeoffTransformer,
    StabilityTransformer,
)
from experiments.core.data import Dataset


def identity_translate(s: str) -> str:
    """Identity translation function for testing."""
    return s


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Create a sample results DataFrame for testing."""
    return pd.DataFrame(
        {
            "model": ["random_forest", "random_forest", "svm", "svm"],
            "technique": ["baseline", "smote", "baseline", "smote"],
            "model_display": ["Random Forest", "Random Forest", "SVM", "SVM"],
            "technique_display": ["Baseline", "SMOTE", "Baseline", "SMOTE"],
            "accuracy_balanced": [0.85, 0.87, 0.82, 0.84],
            "g_mean": [0.84, 0.86, 0.81, 0.83],
            "f1_score": [0.75, 0.78, 0.72, 0.74],
            "precision": [0.80, 0.82, 0.78, 0.80],
            "recall": [0.70, 0.74, 0.67, 0.69],
            "roc_auc": [0.90, 0.92, 0.88, 0.90],
            "seed": [0, 0, 0, 0],
            "best_params": [
                "{'clf__C': 1.0, 'clf__alpha': 0.01}",
                "{'clf__C': 10.0, 'clf__alpha': 0.001}",
                "{'clf__C': 1.0}",
                "{'clf__C': 0.1}",
            ],
        }
    )


class DescribeStabilityTransformer:
    def it_returns_data_with_analysis_type(self, sample_results_df: pd.DataFrame):
        transformer = StabilityTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["analysis_type"] == "stability"

    def it_returns_original_dataframe(self, sample_results_df: pd.DataFrame):
        transformer = StabilityTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["data"].equals(sample_results_df)

    def it_returns_available_metrics(self, sample_results_df: pd.DataFrame):
        transformer = StabilityTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "metrics" in result
        # Should include metrics that exist in the DataFrame
        metric_ids = [m.id for m in result["metrics"]]
        assert "accuracy_balanced" in metric_ids
        assert "g_mean" in metric_ids

    def it_returns_metric_names_mapping(self, sample_results_df: pd.DataFrame):
        transformer = StabilityTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "metric_names" in result
        assert isinstance(result["metric_names"], dict)

    def it_returns_translated_dataset_display(self, sample_results_df: pd.DataFrame):
        def custom_translate(s: str) -> str:
            return f"TR:{s}"

        transformer = StabilityTransformer(custom_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "TR:" in result["dataset_display"]

    def it_uses_custom_metrics_when_provided(self, sample_results_df: pd.DataFrame):
        from experiments.core.analysis.constants import METRIC_ACCURACY_BALANCED

        transformer = StabilityTransformer(identity_translate, metrics=[METRIC_ACCURACY_BALANCED])

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert len(result["metrics"]) == 1
        assert result["metrics"][0].id == "accuracy_balanced"


class DescribeRiskTradeoffTransformer:
    def it_returns_data_with_analysis_type(self, sample_results_df: pd.DataFrame):
        transformer = RiskTradeoffTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["analysis_type"] == "risk_tradeoff"

    def it_aggregates_by_model_and_technique(self, sample_results_df: pd.DataFrame):
        transformer = RiskTradeoffTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert len(df) == 4  # 2 models x 2 techniques

    def it_calculates_mean_precision_recall(self, sample_results_df: pd.DataFrame):
        transformer = RiskTradeoffTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert "precision" in df.columns
        assert "recall" in df.columns

    def it_preserves_display_columns(self, sample_results_df: pd.DataFrame):
        transformer = RiskTradeoffTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert "model_display" in df.columns
        assert "technique_display" in df.columns


class DescribeImbalanceImpactTransformer:
    def it_returns_data_with_analysis_type(self, sample_results_df: pd.DataFrame):
        transformer = ImbalanceImpactTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["analysis_type"] == "imbalance_impact"

    def it_adds_imbalance_ratio_column(self, sample_results_df: pd.DataFrame):
        transformer = ImbalanceImpactTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert "imbalance_ratio" in df.columns

    def it_uses_default_imbalance_ratios(self, sample_results_df: pd.DataFrame):
        transformer = ImbalanceImpactTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        expected_ratio = IMBALANCE_RATIOS[Dataset.TAIWAN_CREDIT.id]
        assert result["imbalance_ratio"] == expected_ratio
        assert all(result["data"]["imbalance_ratio"] == expected_ratio)

    def it_uses_custom_imbalance_ratios(self, sample_results_df: pd.DataFrame):
        custom_ratios = {Dataset.TAIWAN_CREDIT.id: 5.0}
        transformer = ImbalanceImpactTransformer(
            identity_translate, imbalance_ratios=custom_ratios
        )

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["imbalance_ratio"] == 5.0

    def it_returns_available_metrics(self, sample_results_df: pd.DataFrame):
        transformer = ImbalanceImpactTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "metrics" in result
        assert len(result["metrics"]) > 0

    def it_uses_custom_ratio_for_known_dataset(self, sample_results_df: pd.DataFrame):
        custom_ratios: dict[str, float] = {Dataset.TAIWAN_CREDIT.id: 5.0}
        transformer = ImbalanceImpactTransformer(
            identity_translate, imbalance_ratios=custom_ratios
        )

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        # Uses our custom ratio
        assert result["imbalance_ratio"] == 5.0

    def it_defaults_to_ratio_1_when_dataset_not_in_custom_ratios(
        self, sample_results_df: pd.DataFrame
    ):
        # When a dataset is not in the provided ratios dict, it defaults to 1.0
        custom_ratios: dict[str, float] = {"some_other_dataset": 5.0}
        transformer = ImbalanceImpactTransformer(
            identity_translate, imbalance_ratios=custom_ratios
        )

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["imbalance_ratio"] == 1.0


class DescribeCostSensitiveVsResamplingTransformer:
    def it_returns_data_with_analysis_type(self, sample_results_df: pd.DataFrame):
        transformer = CostSensitiveVsResamplingTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["analysis_type"] == "cost_sensitive_vs_resampling"

    def it_returns_original_dataframe(self, sample_results_df: pd.DataFrame):
        transformer = CostSensitiveVsResamplingTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["data"].equals(sample_results_df)

    def it_returns_dataset_display(self, sample_results_df: pd.DataFrame):
        transformer = CostSensitiveVsResamplingTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "dataset_display" in result


class DescribeHyperparameterTransformer:
    def it_returns_data_with_analysis_type(self, sample_results_df: pd.DataFrame):
        transformer = HyperparameterTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["analysis_type"] == "hyperparameter"

    def it_parses_best_params_column(self, sample_results_df: pd.DataFrame):
        transformer = HyperparameterTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert "clf__C" in df.columns

    def it_returns_available_target_params(self, sample_results_df: pd.DataFrame):
        transformer = HyperparameterTransformer(
            identity_translate, target_params=["clf__C", "clf__alpha"]
        )

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        # clf__C is in all rows, clf__alpha only in some
        assert "target_params" in result
        assert "clf__C" in result["target_params"]

    def it_handles_parse_error_gracefully(self):
        df = pd.DataFrame(
            {
                "model": ["rf"],
                "technique": ["baseline"],
                "best_params": ["invalid{json"],
                "accuracy_balanced": [0.85],
            }
        )
        transformer = HyperparameterTransformer(identity_translate)

        result = transformer.transform(df, Dataset.TAIWAN_CREDIT)

        assert "parse_error" in result
        assert result["parse_error"] is not None
        assert result["data"].empty

    def it_returns_none_parse_error_on_success(self, sample_results_df: pd.DataFrame):
        transformer = HyperparameterTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["parse_error"] is None

    def it_preserves_metric_columns(self, sample_results_df: pd.DataFrame):
        transformer = HyperparameterTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert "accuracy_balanced" in df.columns
        assert "g_mean" in df.columns


class DescribeExperimentSummaryTransformer:
    def it_returns_data_with_analysis_type(self, sample_results_df: pd.DataFrame):
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert result["analysis_type"] == "experiment_summary"

    def it_aggregates_mean_and_std(self, sample_results_df: pd.DataFrame):
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        df = result["data"]
        assert "accuracy_balanced_mean" in df.columns
        assert "accuracy_balanced_std" in df.columns

    def it_creates_display_dataframe(self, sample_results_df: pd.DataFrame):
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "display_df" in result
        display_df = result["display_df"]
        assert not display_df.empty

    def it_formats_values_with_mean_and_std(self, sample_results_df: pd.DataFrame):
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        display_df = result["display_df"]
        # Check that values are formatted with ± symbol
        first_metric_col = display_df.columns[2]  # After Model and Technique
        assert "±" in str(display_df[first_metric_col].iloc[0])

    def it_returns_available_metrics(self, sample_results_df: pd.DataFrame):
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "metrics" in result
        assert len(result["metrics"]) > 0

    def it_returns_metric_names_mapping(self, sample_results_df: pd.DataFrame):
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert "metric_names" in result
        assert isinstance(result["metric_names"], dict)

    def it_handles_empty_dataframe(self):
        df = pd.DataFrame(columns=["model", "technique", "model_display", "technique_display"])
        transformer = ExperimentSummaryTransformer(identity_translate)

        result = transformer.transform(df, Dataset.TAIWAN_CREDIT)

        assert result["data"].empty
        assert result["display_df"].empty
        assert result["metrics"] == []

    def it_uses_custom_metrics_when_provided(self, sample_results_df: pd.DataFrame):
        from experiments.core.analysis.constants import METRIC_F1_SCORE

        transformer = ExperimentSummaryTransformer(identity_translate, metrics=[METRIC_F1_SCORE])

        result = transformer.transform(sample_results_df, Dataset.TAIWAN_CREDIT)

        assert len(result["metrics"]) == 1
        assert result["metrics"][0].id == "f1_score"
