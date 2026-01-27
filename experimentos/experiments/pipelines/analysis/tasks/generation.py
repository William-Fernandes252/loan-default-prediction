"""Generation tasks for producing analysis results.

These tasks transform computed metrics into concrete analysis outputs like
summary tables and visualizations. They materialize lazy data and produce
the final result objects.
"""

from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from experiments.core.analysis.metrics import IMBALANCE_RATIOS, Metric
from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines.tasks import TaskResult, TaskStatus
from experiments.pipelines.analysis.pipeline import (
    AnalysisPipelineContext,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)


def generate_summary_table(
    state: AnalysisPipelineState[pl.DataFrame],
    context: AnalysisPipelineContext,
    *,
    technique_filter: Technique | None = None,
) -> AnalysisPipelineTaskResult[pl.DataFrame]:
    """Generate a summary table from computed metrics.

    Filters, sorts, and materializes the metrics LazyFrame into a DataFrame
    suitable for export as a summary table.

    Args:
        state: The current state containing metrics as a LazyFrame.
        context: The pipeline context.
        technique_filter: Optional technique to filter by (e.g., RUS, SMOTE).
            If provided, only rows matching this technique are included.

    Returns:
        AnalysisPipelineTaskResult: Updated state with the materialized
        DataFrame in result_data, or failure status if metrics are missing.
    """
    metrics = state.get("metrics")

    if metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No metrics available to generate summary table.",
        )

    # Build the query with optional filtering
    query = metrics

    if technique_filter is not None:
        query = query.filter(pl.col("technique") == technique_filter.value)

    # Sort by balanced accuracy mean (descending) for ranking
    balanced_accuracy_mean_col = f"{Metric.BALANCED_ACCURACY}_mean"
    query = query.sort(pl.col(balanced_accuracy_mean_col), descending=True)

    # Materialize the result
    result_df: pl.DataFrame = query.collect()

    updated_state: AnalysisPipelineState[pl.DataFrame] = {**state, "result_data": result_df}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Generated summary table with {len(result_df)} rows.",
    )


def create_summary_table_task(
    technique_filter: Technique | None = None,
) -> Any:
    """Factory function to create a summary table generation task with filtering.

    Args:
        technique_filter: Optional technique to filter the results by.

    Returns:
        A task function configured with the specified filter.
    """

    def task(
        state: AnalysisPipelineState[pl.DataFrame],
        context: AnalysisPipelineContext,
    ) -> AnalysisPipelineTaskResult[pl.DataFrame]:
        return generate_summary_table(state, context, technique_filter=technique_filter)

    return task


def generate_tradeoff_plot(
    state: AnalysisPipelineState[plt.Figure],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[plt.Figure]:
    """Generate a precision-sensitivity trade-off scatter plot.

    Creates a Seaborn scatter plot showing the trade-off between precision
    and sensitivity across different techniques and model types.

    Args:
        state: The current state containing metrics as a LazyFrame.
        context: The pipeline context.

    Returns:
        AnalysisPipelineTaskResult: Updated state with the matplotlib Figure
        in result_data, or failure status if metrics are missing.
    """
    metrics = state.get("metrics")

    if metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No metrics available to generate trade-off plot.",
        )

    # Materialize metrics for plotting
    df: pl.DataFrame = metrics.collect()

    # Convert to pandas for seaborn compatibility
    pdf = df.to_pandas()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Column names from the evaluator output
    precision_col = f"{Metric.PRECISION}_mean"
    sensitivity_col = f"{Metric.SENSITIVITY}_mean"

    # Create scatter plot with seaborn
    sns.scatterplot(
        data=pdf,
        x=precision_col,
        y=sensitivity_col,
        hue="technique",
        style="model_type",
        s=100,
        alpha=0.8,
        ax=ax,
    )

    # Styling
    ax.set_xlabel("Precision (Mean)", fontsize=12)
    ax.set_ylabel("Sensitivity / Recall (Mean)", fontsize=12)
    ax.set_title(f"Risk Trade-off: {context.dataset.value}", fontsize=14)
    ax.legend(title="Technique / Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set axis limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    updated_state: AnalysisPipelineState[plt.Figure] = {**state, "result_data": fig}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Generated trade-off plot for dataset {context.dataset.value}.",
    )


# Task aliases for explicit naming in pipelines
GenerateSummaryTableTask = generate_summary_table
"""Task to generate a summary table from metrics."""

GenerateTradeOffPlotTask = generate_tradeoff_plot
"""Task to generate a precision-sensitivity trade-off plot."""


def generate_stability_plot(
    state: AnalysisPipelineState[plt.Figure],
    context: AnalysisPipelineContext,
    *,
    metric: Metric = Metric.BALANCED_ACCURACY,
) -> AnalysisPipelineTaskResult[plt.Figure]:
    """Generate a stability boxplot showing metric distribution across seeds.

    Creates a boxplot visualization showing the variance of a metric across
    different random seeds for each technique and model type combination.

    Args:
        state: The current state containing per_seed_metrics as a LazyFrame.
        context: The pipeline context.
        metric: The metric to visualize. Defaults to balanced accuracy.

    Returns:
        AnalysisPipelineTaskResult: Updated state with the matplotlib Figure
        in result_data, or failure status if metrics are missing.
    """
    per_seed_metrics = state.get("per_seed_metrics")

    if per_seed_metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No per-seed metrics available to generate stability plot.",
        )

    # Materialize metrics for plotting
    df: pl.DataFrame = per_seed_metrics.collect()

    # Convert to pandas for seaborn compatibility
    pdf = df.to_pandas()

    # Sort technique and model_type alphabetically for consistent ordering
    pdf = pdf.sort_values(["technique", "model_type"])

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create boxplot with seaborn
    sns.boxplot(
        data=pdf,
        x="technique",
        y=metric.value,
        hue="model_type",
        palette="viridis",
        showfliers=False,
        linewidth=1.5,
        ax=ax,
    )

    # Overlay with stripplot for individual points
    sns.stripplot(
        data=pdf,
        x="technique",
        y=metric.value,
        hue="model_type",
        dodge=True,
        alpha=0.4,
        palette="dark:black",
        legend=False,
        ax=ax,
        size=3,
    )

    # Styling
    metric_display = metric.value.replace("_", " ").title()
    ax.set_ylabel(metric_display, fontsize=12)
    ax.set_xlabel("Handling Technique", fontsize=12)
    ax.set_title(f"Stability Analysis: {metric_display} - {context.dataset.value}", fontsize=14)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    updated_state: AnalysisPipelineState[plt.Figure] = {**state, "result_data": fig}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Generated stability plot for {metric.value}.",
    )


def create_stability_plot_task(metric: Metric = Metric.BALANCED_ACCURACY) -> Any:
    """Factory function to create a stability plot task with a specific metric.

    Args:
        metric: The metric to visualize.

    Returns:
        A task function configured with the specified metric.
    """

    def task(
        state: AnalysisPipelineState[plt.Figure],
        context: AnalysisPipelineContext,
    ) -> AnalysisPipelineTaskResult[plt.Figure]:
        return generate_stability_plot(state, context, metric=metric)

    return task


def generate_imbalance_impact_plot(
    state: AnalysisPipelineState[plt.Figure],
    context: AnalysisPipelineContext,
    *,
    metric: Metric = Metric.BALANCED_ACCURACY,
) -> AnalysisPipelineTaskResult[plt.Figure]:
    """Generate a scatter plot showing metric vs imbalance ratio.

    Creates a visualization showing how the imbalance ratio affects
    model performance. This is a single-dataset analysis that shows
    the data point for the current dataset.

    Args:
        state: The current state containing metrics as a LazyFrame.
        context: The pipeline context.
        metric: The metric to visualize. Defaults to balanced accuracy.

    Returns:
        AnalysisPipelineTaskResult: Updated state with the matplotlib Figure
        in result_data, or failure status if metrics are missing.
    """
    metrics = state.get("metrics")

    if metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No metrics available to generate imbalance impact plot.",
        )

    # Materialize metrics for plotting
    df: pl.DataFrame = metrics.collect()

    # Get imbalance ratio for current dataset
    imbalance_ratio = IMBALANCE_RATIOS.get(context.dataset, 1.0)

    # Add imbalance ratio column
    df = df.with_columns(pl.lit(imbalance_ratio).alias("imbalance_ratio"))

    # Sort by technique and model_type for consistent ordering
    df = df.sort("technique", "model_type")

    # Convert to pandas for seaborn compatibility
    pdf = df.to_pandas()

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Metric column name (mean)
    metric_col = f"{metric.value}_mean"

    # Create scatter plot
    sns.scatterplot(
        data=pdf,
        x="imbalance_ratio",
        y=metric_col,
        hue="technique",
        style="model_type",
        s=100,
        palette="muted",
        alpha=0.7,
        edgecolor="k",
        ax=ax,
    )

    # Use log scale for imbalance ratio
    ax.set_xscale("log")

    # Styling
    metric_display = metric.value.replace("_", " ").title()
    ax.set_xlabel("Imbalance Ratio (Majority/Minority) - Log Scale", fontsize=12)
    ax.set_ylabel(f"{metric_display} (Mean)", fontsize=12)
    ax.set_title(
        f"{metric_display} vs. Imbalance Ratio - {context.dataset.value}",
        fontsize=14,
    )
    ax.legend(title="Technique / Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    updated_state: AnalysisPipelineState[plt.Figure] = {**state, "result_data": fig}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Generated imbalance impact plot for {context.dataset.value}.",
    )


def create_imbalance_impact_task(metric: Metric = Metric.BALANCED_ACCURACY) -> Any:
    """Factory function to create an imbalance impact task with a specific metric.

    Args:
        metric: The metric to visualize.

    Returns:
        A task function configured with the specified metric.
    """

    def task(
        state: AnalysisPipelineState[plt.Figure],
        context: AnalysisPipelineContext,
    ) -> AnalysisPipelineTaskResult[plt.Figure]:
        return generate_imbalance_impact_plot(state, context, metric=metric)

    return task


# Cost-sensitive techniques for comparison analysis
_COST_SENSITIVE_TECHNIQUES = {Technique.META_COST, Technique.CS_SVM}
_RESAMPLING_TECHNIQUES = {
    Technique.SMOTE,
    Technique.RANDOM_UNDER_SAMPLING,
    Technique.SMOTE_TOMEK,
}


def generate_cs_vs_resampling_plot(
    state: AnalysisPipelineState[plt.Figure],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[plt.Figure]:
    """Generate a bar plot comparing cost-sensitive vs resampling techniques.

    Creates a grouped bar chart showing balanced accuracy performance for
    cost-sensitive methods (MetaCost, CS-SVM) vs resampling methods
    (SMOTE, RUS, SMOTE-Tomek).

    Args:
        state: The current state containing per_seed_metrics as a LazyFrame.
        context: The pipeline context.

    Returns:
        AnalysisPipelineTaskResult: Updated state with the matplotlib Figure
        in result_data, or failure status if metrics are missing.
    """
    per_seed_metrics = state.get("per_seed_metrics")

    if per_seed_metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No per-seed metrics available to generate comparison plot.",
        )

    # Materialize metrics for plotting
    df: pl.DataFrame = per_seed_metrics.collect()

    # Filter to relevant techniques
    relevant_techniques = _COST_SENSITIVE_TECHNIQUES | _RESAMPLING_TECHNIQUES
    technique_values = [t.value for t in relevant_techniques]
    df = df.filter(pl.col("technique").is_in(technique_values))

    # Sort by technique and model_type for consistent ordering
    df = df.sort("technique", "model_type")

    # Convert to pandas for seaborn compatibility
    pdf = df.to_pandas()

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar plot with error bars
    sns.barplot(
        data=pdf,
        x="technique",
        y=Metric.BALANCED_ACCURACY.value,
        hue="model_type",
        palette="Set2",
        errorbar="sd",
        capsize=0.1,
        ax=ax,
    )

    # Styling
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_xlabel("Technique", fontsize=12)
    ax.set_title(
        f"Cost-Sensitive vs Resampling Performance - {context.dataset.value}",
        fontsize=14,
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    updated_state: AnalysisPipelineState[plt.Figure] = {**state, "result_data": fig}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Generated CS vs resampling plot for {context.dataset.value}.",
    )


def generate_metrics_heatmap(
    state: AnalysisPipelineState[plt.Figure],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[plt.Figure]:
    """Generate a heatmap showing all metrics across techniques and models.

    Creates a heatmap visualization with techniques/models on one axis and
    metrics on the other, showing mean performance values. Rows and columns
    are sorted alphabetically.

    Args:
        state: The current state containing metrics as a LazyFrame.
        context: The pipeline context.

    Returns:
        AnalysisPipelineTaskResult: Updated state with the matplotlib Figure
        in result_data, or failure status if metrics are missing.
    """
    metrics = state.get("metrics")

    if metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No metrics available to generate heatmap.",
        )

    # Materialize metrics for plotting
    df: pl.DataFrame = metrics.collect()

    # Sort by technique and model_type for consistent ordering
    df = df.sort("technique", "model_type")

    # Create a combined index column for rows
    df = df.with_columns(
        (pl.col("technique") + " / " + pl.col("model_type")).alias("technique_model")
    )

    # Select mean columns for heatmap
    metric_cols = [f"{m.value}_mean" for m in Metric]
    available_cols = [col for col in metric_cols if col in df.columns]

    # Pivot to create heatmap matrix
    pivot_df = df.select(["technique_model"] + available_cols)

    # Convert to pandas for seaborn
    pdf = pivot_df.to_pandas().set_index("technique_model")

    # Sort index alphabetically
    pdf = pdf.sort_index()

    # Rename columns for display (remove _mean suffix)
    pdf.columns = [col.replace("_mean", "").replace("_", " ").title() for col in pdf.columns]

    # Sort columns alphabetically
    pdf = pdf[sorted(pdf.columns)]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, max(8.0, len(pdf) * 0.5)))

    # Create heatmap
    sns.heatmap(
        pdf,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Score"},
    )

    # Styling
    ax.set_title(f"Metrics Heatmap - {context.dataset.value}", fontsize=14)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Technique / Model", fontsize=12)

    plt.tight_layout()

    updated_state: AnalysisPipelineState[plt.Figure] = {**state, "result_data": fig}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Generated metrics heatmap for {context.dataset.value}.",
    )


# Additional task aliases
GenerateStabilityPlotTask = generate_stability_plot
"""Task to generate a stability boxplot from per-seed metrics."""

GenerateImbalanceImpactPlotTask = generate_imbalance_impact_plot
"""Task to generate an imbalance impact scatter plot."""

GenerateCsVsResamplingPlotTask = generate_cs_vs_resampling_plot
"""Task to generate a cost-sensitive vs resampling comparison plot."""

GenerateMetricsHeatmapTask = generate_metrics_heatmap
"""Task to generate a metrics heatmap."""


__all__ = [
    "GenerateSummaryTableTask",
    "GenerateTradeOffPlotTask",
    "GenerateStabilityPlotTask",
    "GenerateImbalanceImpactPlotTask",
    "GenerateCsVsResamplingPlotTask",
    "GenerateMetricsHeatmapTask",
    "generate_summary_table",
    "generate_tradeoff_plot",
    "generate_stability_plot",
    "generate_imbalance_impact_plot",
    "generate_cs_vs_resampling_plot",
    "generate_metrics_heatmap",
    "create_summary_table_task",
    "create_stability_plot_task",
    "create_imbalance_impact_task",
]
