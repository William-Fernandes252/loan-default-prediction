import matplotlib.pyplot as plt
import polars as pl

from experiments.core.analysis.metrics import Metric
from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines import TaskResult, TaskStatus
from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineFactory,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)
from experiments.pipelines.analysis.tasks.common import compute_per_seed_metrics
from experiments.pipelines.analysis.tasks.generation import generate_cs_vs_resampling_plot

# The result is a Figure comparing cost-sensitive vs resampling methods
type CostSensitiveVsResamplingPlot = plt.Figure


def perform_cost_sensitive_vs_resampling_comparison(
    state: AnalysisPipelineState,
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult:
    """Performs cost-sensitive vs resampling comparison on the model predictions.

    This task filters the computed per-seed metrics to isolate Cost-Sensitive techniques
    (MetaCost, Cost-Sensitive SVM) and Resampling techniques (SMOTE, RUS,
    SMOTE-Tomek), then aggregates and selects the best performing technique from
    each group per model type for comparison.

    Args:
        state: The current state containing "per_seed_metrics" (pl.LazyFrame).
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult: Updated state with "result_data" containing
        the comparison DataFrame.
    """
    per_seed_metrics: pl.LazyFrame | None = state.get("per_seed_metrics")
    if per_seed_metrics is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "Per-seed metrics not found in state. Ensure compute_per_seed_metrics was run.",
        )

    # 1. Define Groups
    cs_techniques = [Technique.META_COST, Technique.CS_SVM]
    resampling_techniques = [
        Technique.SMOTE,
        Technique.RANDOM_UNDER_SAMPLING,
        Technique.SMOTE_TOMEK,
    ]

    # 2. Aggregate per-seed metrics by model_type and technique (mean)
    aggregated_lf = per_seed_metrics.group_by("model_type", "technique").agg(
        pl.col(Metric.G_MEAN).mean().alias("g_mean"),
        pl.col(Metric.F1_SCORE).mean().alias("f1_score"),
        pl.col(Metric.BALANCED_ACCURACY).mean().alias("balanced_accuracy"),
    )

    # 3. Filter Metrics for relevant techniques
    comparison_lf = aggregated_lf.filter(
        pl.col("technique").is_in(cs_techniques + resampling_techniques)
    )

    # 4. Categorize into Groups (Cost-Sensitive vs Resampling)
    # We create a new column 'approach' to group the techniques
    grouped_lf = comparison_lf.with_columns(
        pl.when(pl.col("technique").is_in(cs_techniques))
        .then(pl.lit("Cost-Sensitive"))
        .otherwise(pl.lit("Resampling"))
        .alias("approach")
    )

    # 5. Find the Best Performing Technique per Group and Model
    # For a fair comparison, we take the best technique within each group for each model.
    # e.g., for Random Forest, we compare (Best of SMOTE/RUS) vs (MetaCost)
    best_per_group_lf = (
        grouped_lf.sort("g_mean", descending=True)
        .group_by("model_type", "approach")
        .first()
        .sort("model_type", "approach")
    )

    # 6. Collect results
    # The result_data will be a DataFrame suitable for generating the LaTeX table
    result_df = best_per_group_lf.collect()

    state["result_data"] = result_df

    return TaskResult(
        state, TaskStatus.SUCCESS, "Performed Cost-Sensitive vs Resampling comparison."
    )


class CostSensitiveVsResamplingComparisonPipelineFactory(
    AnalysisPipelineFactory[CostSensitiveVsResamplingPlot]
):
    """Factory for creating cost-sensitive vs resampling comparison pipelines."""

    _NAME = "CostSensitiveVsResamplingComparison"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[CostSensitiveVsResamplingPlot],
    ) -> None:
        # Step 1: Compute per-seed metrics (needed for both comparison and plot)
        pipeline.add_step(
            name="ComputePerSeedMetrics",
            task=compute_per_seed_metrics,
        )

        # Step 2: Perform the specific comparison logic (aggregates per-seed metrics)
        pipeline.add_step(
            name="PerformCostSensitiveVsResamplingComparison",
            task=perform_cost_sensitive_vs_resampling_comparison,
        )

        # Step 3: Generate the comparison plot
        pipeline.add_step(
            name="GenerateCostSensitiveVsResamplingPlot",
            task=generate_cs_vs_resampling_plot,  # type: ignore[arg-type]
        )
