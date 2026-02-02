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
from experiments.pipelines.analysis.tasks.common import compute_metrics
from experiments.pipelines.analysis.tasks.generation import create_imbalance_impact_task

# The result is a Figure showing imbalance impact
type ImbalanceImpactPlot = plt.Figure


def perform_imbalance_impact_analysis(
    state: AnalysisPipelineState, context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult:
    """Performs imbalance impact analysis on the model predictions.

    This analysis quantifies the impact of class imbalance by comparing the
    performance of the Baseline model (no treatment) against the Best Performing
    model (using any imbalance handling technique).

    The 'Impact' is defined as the potential performance gain (Improvement)
    realized by addressing the imbalance.

    Args:
        state: The current state containing "metrics" (pl.LazyFrame).
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult: Updated state with "result_data" containing
        the comparison DataFrame.
    """
    metrics_lf: pl.LazyFrame | None = state.get("metrics")
    if metrics_lf is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "Metrics not found in state. Ensure compute_analysis_metrics was run.",
        )

    # 1. Isolate the Baseline Metrics
    # We select G-Mean (primary) and Balanced Accuracy (secondary)
    baseline_lf = metrics_lf.filter(pl.col("technique") == Technique.BASELINE).select(
        pl.col("model_type"),
        pl.col(f"{Metric.G_MEAN}_mean").alias("baseline_g_mean"),
        pl.col(f"{Metric.BALANCED_ACCURACY}_mean").alias("baseline_balanced_acc"),
    )

    # 2. Identify the Best Performing Technique (Benchmark)
    # We look at ALL non-baseline techniques to find the upper bound of performance
    # recoverable for this specific dataset.
    best_treated_lf = (
        metrics_lf.filter(pl.col("technique") != Technique.BASELINE)
        .sort(f"{Metric.G_MEAN}_mean", descending=True)
        .group_by("model_type")
        .first()
        .select(
            pl.col("model_type"),
            pl.col("technique").alias("best_technique"),
            pl.col(f"{Metric.G_MEAN}_mean").alias("best_g_mean"),
            pl.col(f"{Metric.BALANCED_ACCURACY}_mean").alias("best_balanced_acc"),
        )
    )

    # 3. Join and Calculate Impact (Improvement)
    impact_lf = (
        baseline_lf.join(best_treated_lf, on="model_type", how="left")
        .with_columns(
            (pl.col("best_g_mean") - pl.col("baseline_g_mean")).alias("g_mean_improvement"),
            (pl.col("best_balanced_acc") - pl.col("baseline_balanced_acc")).alias(
                "balanced_acc_improvement"
            ),
        )
        .sort("g_mean_improvement", descending=True)
    )

    # Collect result
    state["result_data"] = impact_lf.collect()

    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        "Performed Imbalance Impact analysis.",
    )


class ImbalanceImpactAnalysisPipelineFactory(AnalysisPipelineFactory[ImbalanceImpactPlot]):
    """Factory for creating imbalance impact analysis pipelines."""

    _NAME = "ImbalanceImpactAnalysis"

    def __init__(
        self,
        analysis_artifacts_repository,
        metric: Metric = Metric.BALANCED_ACCURACY,
    ) -> None:
        super().__init__(analysis_artifacts_repository)
        self.metric = metric

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[ImbalanceImpactPlot],
    ) -> None:
        # Step 1: Compute metrics from raw predictions
        pipeline.add_step(
            name="ComputeMetrics",
            task=compute_metrics,
        )

        # Step 2: Compare Baseline vs. Best Treated
        pipeline.add_step(
            name="PerformImbalanceImpactAnalysis",
            task=perform_imbalance_impact_analysis,
        )

        # Step 3: Generate imbalance impact plot with specified metric
        pipeline.add_step(
            name="GenerateImbalanceImpactPlot",
            task=create_imbalance_impact_task(metric=self.metric),
        )
