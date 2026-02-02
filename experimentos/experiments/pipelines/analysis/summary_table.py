import polars as pl

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
from experiments.pipelines.analysis.tasks.generation import create_summary_table_task

type SummaryTable = pl.DataFrame
"""Represents the summary table analysis result as a DataFrame."""


def perform_summary_table_analysis(
    state: AnalysisPipelineState, context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult:
    """Performs summary table analysis on the model results.

    This task formats the computed metrics into a summary table suitable
    for export as a LaTeX table, sorted by balanced accuracy.

    Args:
        state: The current state containing "metrics" (pl.LazyFrame).
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult: Updated state with "result_data" containing
        the summary table DataFrame.
    """
    metrics_lf: pl.LazyFrame | None = state.get("metrics")
    if metrics_lf is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "Metrics not found in state. Ensure compute_metrics was run.",
        )

    # The task already handles filtering and sorting
    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        "Performed Summary Table analysis.",
    )


class SummaryTablePipelineFactory(AnalysisPipelineFactory[SummaryTable]):
    """Factory for creating summary table analysis pipelines."""

    _NAME = "SummaryTable"

    def __init__(
        self,
        analysis_artifacts_repository,
        technique_filter: Technique | None = None,
    ) -> None:
        super().__init__(analysis_artifacts_repository)
        self.technique_filter = technique_filter

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[SummaryTable],
    ) -> None:
        # Step 1: Compute metrics from raw predictions
        pipeline.add_step(
            name="ComputeMetrics",
            task=compute_metrics,
        )

        # Step 2: Generate summary table with optional technique filtering
        pipeline.add_step(
            name="GenerateSummaryTable",
            task=create_summary_table_task(technique_filter=self.technique_filter),
        )
