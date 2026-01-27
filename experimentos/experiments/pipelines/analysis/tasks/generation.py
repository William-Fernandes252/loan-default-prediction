"""Generation tasks for producing analysis results.

These tasks transform computed metrics into concrete analysis outputs like
summary tables and visualizations. They materialize lazy data and produce
the final result objects.
"""

from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from experiments.core.analysis.metrics import Metric
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
    balanced_accuracy_mean_col = f"{Metric.ACCURACY_BALANCED}_mean"
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


__all__ = [
    "GenerateSummaryTableTask",
    "GenerateTradeOffPlotTask",
    "generate_summary_table",
    "generate_tradeoff_plot",
    "create_summary_table_task",
]
