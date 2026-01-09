"""Data transformer implementations for the analysis pipeline.

This module provides concrete implementations of the DataTransformer protocol
for various analysis types (stability, risk tradeoff, etc.).
"""

from abc import ABC, abstractmethod
import ast
from typing import Any

import pandas as pd

from experiments.core.analysis.metrics import (
    IMBALANCE_METRICS,
    IMBALANCE_RATIOS,
    STABILITY_METRICS,
    SUMMARY_METRICS,
    MetricConfig,
    get_metric_display_names,
)
from experiments.core.analysis.protocols import TranslationFunc
from experiments.core.data import Dataset


class BaseTransformer(ABC):
    """Base class for data transformers with common functionality.

    Provides shared utilities for translation and metric handling.
    """

    def __init__(self, translate: TranslationFunc) -> None:
        """Initialize the transformer.

        Args:
            translate: Translation function for display names.
        """
        self._translate = translate

    def _get_dataset_display(self, dataset: Dataset) -> str:
        """Get translated display name for a dataset."""
        try:
            return self._translate(dataset.display_name)
        except (ValueError, AttributeError):
            return dataset.id

    @abstractmethod
    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform the input DataFrame."""
        ...


class StabilityTransformer(BaseTransformer):
    """Transforms data for stability and variance analysis.

    Prepares data for boxplot visualization showing performance spread
    across random seeds.
    """

    def __init__(
        self,
        translate: TranslationFunc,
        metrics: list[MetricConfig] | None = None,
    ) -> None:
        """Initialize the transformer.

        Args:
            translate: Translation function.
            metrics: List of metrics to include. Defaults to STABILITY_METRICS.
        """
        super().__init__(translate)
        self._metrics = metrics or STABILITY_METRICS

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform data for stability analysis.

        Returns:
            Dictionary containing:
                - 'data': The input DataFrame (no aggregation needed for boxplots)
                - 'metrics': List of available metric configs
                - 'metric_names': Mapping of metric ID to display name
                - 'dataset_display': Translated dataset name
        """
        available_metrics = [m for m in self._metrics if m.id in df.columns]
        metric_names = get_metric_display_names(
            self._translate,
            [m.id for m in available_metrics],
        )

        return {
            "data": df,
            "metrics": available_metrics,
            "metric_names": metric_names,
            "dataset_display": self._get_dataset_display(dataset),
            "analysis_type": "stability",
        }


class RiskTradeoffTransformer(BaseTransformer):
    """Transforms data for precision-recall tradeoff analysis.

    Aggregates data by model and technique to show mean precision/recall
    for scatter plot visualization.
    """

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform data for risk tradeoff analysis.

        Returns:
            Dictionary containing:
                - 'data': Aggregated DataFrame with mean precision/recall
                - 'dataset_display': Translated dataset name
        """
        # Calculate mean performance across seeds
        df_agg = df.groupby(["model", "technique"])[["precision", "recall"]].mean().reset_index()

        # Apply translations
        df_agg["model_display"] = df_agg["model"].apply(lambda x: self._translate(x) if x else x)
        df_agg["technique_display"] = df_agg["technique"].apply(
            lambda x: self._translate(x) if x else x
        )

        # Copy display values from original if available
        if "model_display" in df.columns:
            model_map = df.drop_duplicates("model").set_index("model")["model_display"]
            df_agg["model_display"] = df_agg["model"].map(model_map)

        if "technique_display" in df.columns:
            tech_map = df.drop_duplicates("technique").set_index("technique")["technique_display"]
            df_agg["technique_display"] = df_agg["technique"].map(tech_map)

        return {
            "data": df_agg,
            "dataset_display": self._get_dataset_display(dataset),
            "analysis_type": "risk_tradeoff",
        }


class ImbalanceImpactTransformer(BaseTransformer):
    """Transforms data for imbalance impact analysis.

    Injects imbalance ratio and prepares data for scatter plots
    showing metrics vs. imbalance ratio.
    """

    def __init__(
        self,
        translate: TranslationFunc,
        metrics: list[MetricConfig] | None = None,
        imbalance_ratios: dict[str, float] | None = None,
    ) -> None:
        """Initialize the transformer.

        Args:
            translate: Translation function.
            metrics: List of metrics to analyze. Defaults to IMBALANCE_METRICS.
            imbalance_ratios: Custom imbalance ratios. Defaults to IMBALANCE_RATIOS.
        """
        super().__init__(translate)
        self._metrics = metrics or IMBALANCE_METRICS
        self._ratios = imbalance_ratios or IMBALANCE_RATIOS

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform data for imbalance impact analysis.

        Returns:
            Dictionary containing:
                - 'data': DataFrame with imbalance_ratio column added
                - 'metrics': List of available metric configs
                - 'metric_names': Mapping of metric ID to display name
                - 'dataset_display': Translated dataset name
        """
        ratio = self._ratios.get(dataset.id, 1.0)
        df_plot = df.copy()
        df_plot["imbalance_ratio"] = ratio

        available_metrics = [m for m in self._metrics if m.id in df_plot.columns]
        metric_names = get_metric_display_names(
            self._translate,
            [m.id for m in available_metrics],
        )

        return {
            "data": df_plot,
            "metrics": available_metrics,
            "metric_names": metric_names,
            "dataset_display": self._get_dataset_display(dataset),
            "imbalance_ratio": ratio,
            "analysis_type": "imbalance_impact",
        }


class CostSensitiveVsResamplingTransformer(BaseTransformer):
    """Transforms data for cost-sensitive vs resampling comparison.

    Prepares data for bar plot visualization comparing techniques.
    """

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform data for cost-sensitive vs resampling analysis.

        Returns:
            Dictionary containing:
                - 'data': The input DataFrame (no transformation needed)
                - 'dataset_display': Translated dataset name
        """
        return {
            "data": df,
            "dataset_display": self._get_dataset_display(dataset),
            "analysis_type": "cost_sensitive_vs_resampling",
        }


class HyperparameterTransformer(BaseTransformer):
    """Transforms data for hyperparameter effects analysis.

    Parses the best_params column and merges with metrics for
    visualizing hyperparameter impact.
    """

    def __init__(
        self,
        translate: TranslationFunc,
        target_params: list[str] | None = None,
    ) -> None:
        """Initialize the transformer.

        Args:
            translate: Translation function.
            target_params: List of hyperparameters to analyze.
        """
        super().__init__(translate)
        self._target_params = target_params or [
            "clf__alpha",
            "clf__C",
            "clf__learning_rate",
        ]

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform data for hyperparameter analysis.

        Returns:
            Dictionary containing:
                - 'data': DataFrame with parsed hyperparameters merged
                - 'target_params': List of available target parameters
                - 'dataset_display': Translated dataset name
                - 'parse_error': Error message if parsing failed, None otherwise
        """
        # Safely parse the 'best_params' string into columns
        try:
            hp_df = df["best_params"].apply(lambda x: pd.Series(ast.literal_eval(x)))
        except (ValueError, SyntaxError, KeyError) as e:
            return {
                "data": pd.DataFrame(),
                "target_params": [],
                "dataset_display": self._get_dataset_display(dataset),
                "parse_error": str(e),
                "analysis_type": "hyperparameter",
            }

        # Combine metrics with extracted hyperparameters
        metric_columns = [
            col
            for col in (
                "accuracy_balanced",
                "g_mean",
                "f1_score",
                "precision",
                "recall",
            )
            if col in df.columns
        ]

        display_cols = [col for col in ("technique_display", "model_display") if col in df.columns]
        meta_columns = ["technique", "model", *display_cols]

        merged_df = pd.concat(
            [
                df[meta_columns + metric_columns].reset_index(drop=True),
                hp_df.reset_index(drop=True),
            ],
            axis=1,
        )

        # Filter to available target params
        available_params = [p for p in self._target_params if p in merged_df.columns]

        return {
            "data": merged_df,
            "target_params": available_params,
            "dataset_display": self._get_dataset_display(dataset),
            "parse_error": None,
            "analysis_type": "hyperparameter",
        }


class ExperimentSummaryTransformer(BaseTransformer):
    """Transforms data for tabular experiment summary.

    Aggregates results by model and technique, computing mean and std
    for each metric.
    """

    def __init__(
        self,
        translate: TranslationFunc,
        metrics: list[MetricConfig] | None = None,
    ) -> None:
        """Initialize the transformer.

        Args:
            translate: Translation function.
            metrics: List of metrics to summarize. Defaults to SUMMARY_METRICS.
        """
        super().__init__(translate)
        self._metrics = metrics or SUMMARY_METRICS

    def transform(self, df: pd.DataFrame, dataset: Dataset) -> dict[str, Any]:
        """Transform data for experiment summary.

        Returns:
            Dictionary containing:
                - 'data': DataFrame with aggregated statistics
                - 'display_df': Formatted DataFrame for display
                - 'metrics': Available metric configs
                - 'metric_names': Mapping of metric ID to display name
                - 'dataset_display': Translated dataset name
        """
        available_metrics = [m for m in self._metrics if m.id in df.columns]
        metric_ids = [m.id for m in available_metrics]

        if not available_metrics:
            return {
                "data": pd.DataFrame(),
                "display_df": pd.DataFrame(),
                "metrics": [],
                "metric_names": {},
                "dataset_display": self._get_dataset_display(dataset),
                "analysis_type": "experiment_summary",
            }

        group_cols = ["model", "technique", "model_display", "technique_display"]

        # Calculate mean and std
        agg_dict = {m: ["mean", "std"] for m in metric_ids}
        summary: pd.DataFrame = df.groupby(group_cols)[metric_ids].agg(agg_dict)

        # Flatten MultiIndex columns
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.reset_index()

        # Create display DataFrame
        metric_names = get_metric_display_names(self._translate, metric_ids)

        display_df = pd.DataFrame()
        display_df[self._translate("Model")] = summary["model_display"]
        display_df[self._translate("Technique")] = summary["technique_display"]

        for m in available_metrics:
            m_display = metric_names.get(m.id, m.id)
            mean_col = f"{m.id}_mean"
            std_col = f"{m.id}_std"
            display_df[m_display] = summary.apply(
                lambda row, mc=mean_col, sc=std_col: f"{row[mc]:.4f} ± {row[sc]:.4f}",
                axis=1,
            )

        # Sort by Model name
        model_col = self._translate("Model")
        display_df = display_df.sort_values(by=model_col)

        return {
            "data": summary,
            "display_df": display_df,
            "metrics": available_metrics,
            "metric_names": metric_names,
            "dataset_display": self._get_dataset_display(dataset),
            "analysis_type": "experiment_summary",
        }


__all__ = [
    "BaseTransformer",
    "StabilityTransformer",
    "RiskTradeoffTransformer",
    "ImbalanceImpactTransformer",
    "CostSensitiveVsResamplingTransformer",
    "HyperparameterTransformer",
    "ExperimentSummaryTransformer",
]
