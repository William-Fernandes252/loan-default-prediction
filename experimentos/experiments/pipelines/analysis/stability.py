import matplotlib.pyplot as plt
import polars as pl

from experiments.core.analysis.metrics import Metric
from experiments.lib.pipelines import TaskResult, TaskStatus
from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineFactory,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)
from experiments.pipelines.analysis.tasks.common import compute_per_seed_metrics
from experiments.pipelines.analysis.tasks.generation import create_stability_plot_task

# The result is a Figure generated from seed-level metrics
type StabilityPlot = plt.Figure


def perform_stability_analysis(
    state: AnalysisPipelineState, context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult:
    """Performs stability analysis processing.

    This task prepares the seed-level metrics for visualization. It ensures
    the data is correctly formatted and sorted to generate boxplots that
    show the spread of performance across the 30 iterations.

    Args:
        state: The current state containing "metrics_per_seed".
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult: Updated state with "result_data" containing
        the Stability DataFrame.
    """
    metrics_lf: pl.LazyFrame | None = state.get("per_seed_metrics")
    if metrics_lf is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "Seed-level metrics not found. Ensure compute_seed_level_metrics was run.",
        )

    # We collect the full dataset. The plotting logic (in the artifact generator)
    # will handle the selection of specific metrics (e.g., Balanced Accuracy)
    # for the boxplots.
    # We sort by model_type and technique to ensure consistent plotting order.
    stability_lf = metrics_lf.sort(["model_type", "technique", "seed"])

    state["result_data"] = stability_lf.collect()

    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        "Performed Stability analysis.",
    )


class StabilityAnalysisPipelineFactory(AnalysisPipelineFactory[StabilityPlot]):
    """Factory for creating stability analysis pipelines."""

    _NAME = "StabilityAnalysis"

    def __init__(
        self,
        analysis_artifacts_repository,
        metric: Metric = Metric.BALANCED_ACCURACY,
    ) -> None:
        super().__init__(analysis_artifacts_repository)
        self.metric = metric

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[StabilityPlot],
    ) -> None:
        # Step 1: Compute raw metrics for every seed (Specific to Stability)
        # Note: We do NOT use the standard compute_analysis_metrics here because
        # we need the raw distribution, not the aggregates.
        pipeline.add_step(
            name="ComputeSeedMetrics",
            task=compute_per_seed_metrics,
        )

        # Step 2: Format data for the report
        pipeline.add_step(
            name="PerformStabilityAnalysis",
            task=perform_stability_analysis,
        )

        # Step 3: Generate stability plot with specified metric
        pipeline.add_step(
            name="GenerateStabilityPlot",
            task=create_stability_plot_task(metric=self.metric),
        )
