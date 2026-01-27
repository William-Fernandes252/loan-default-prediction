"""Factory functions for building analysis pipelines.

These factories construct analysis pipelines with the core business logic steps
(Load -> Compute -> Generate). Persistence steps are intentionally excluded
and should be appended at runtime by the CLI layer.

This follows the "Append Strategy" pattern where:
- Factories build the pure data transformation pipeline
- CLI/orchestration layer appends IO side-effects (persistence)
"""

import matplotlib.pyplot as plt
import polars as pl

from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines import Pipeline
from experiments.pipelines.analysis.pipeline import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineState,
)
from experiments.pipelines.analysis.tasks.common import (
    compute_metrics,
    load_experiment_results,
)
from experiments.pipelines.analysis.tasks.generation import (
    create_summary_table_task,
    generate_tradeoff_plot,
)


def build_summary_table_pipeline(
    technique: Technique | None = None,
) -> AnalysisPipeline[pl.DataFrame]:
    """Build a pipeline for generating summary tables from experiment results.

    Creates a pipeline with the following steps:
    1. LoadExperimentResults: Fetches predictions from repository
    2. ComputeMetrics: Computes evaluation metrics (lazy)
    3. GenerateSummaryTable: Filters, sorts, and materializes the table

    Persistence steps should be appended by the caller using `pipeline.add_step()`.

    Args:
        technique: Optional technique to filter results by.
            If provided, only results for this technique are included.

    Returns:
        AnalysisPipeline[pl.DataFrame]: A configured pipeline that produces
        a Polars DataFrame in the `result_data` state key.

    Example:
        >>> pipeline = build_summary_table_pipeline(technique=Technique.SMOTE)
        >>> pipeline.add_step("SaveArtifact", SaveTableArtifactTask)
        >>> executor.execute(pipeline, initial_state={}, context=ctx)
    """
    # Build pipeline name based on filter
    name_suffix = f"_{technique.value}" if technique else "_all"
    pipeline_name = f"SummaryTable{name_suffix}"

    pipeline: AnalysisPipeline[pl.DataFrame] = Pipeline[
        AnalysisPipelineState[pl.DataFrame], AnalysisPipelineContext
    ](name=pipeline_name)

    # Step 1: Load experiment predictions
    pipeline.add_step(
        name="LoadExperimentResults",
        task=load_experiment_results,
    )

    # Step 2: Compute evaluation metrics (kept as LazyFrame)
    pipeline.add_step(
        name="ComputeMetrics",
        task=compute_metrics,
    )

    # Step 3: Generate the summary table (materializes the DataFrame)
    pipeline.add_step(
        name="GenerateSummaryTable",
        task=create_summary_table_task(technique_filter=technique),
    )

    return pipeline


def build_tradeoff_plot_pipeline() -> AnalysisPipeline[plt.Figure]:
    """Build a pipeline for generating precision-sensitivity trade-off plots.

    Creates a pipeline with the following steps:
    1. LoadExperimentResults: Fetches predictions from repository
    2. ComputeMetrics: Computes evaluation metrics (lazy)
    3. GenerateTradeOffPlot: Creates the scatter plot visualization

    Persistence steps should be appended by the caller using `pipeline.add_step()`.

    Returns:
        AnalysisPipeline[plt.Figure]: A configured pipeline that produces
        a matplotlib Figure in the `result_data` state key.

    Example:
        >>> pipeline = build_tradeoff_plot_pipeline()
        >>> pipeline.add_step("SaveArtifact", SaveFigureArtifactTask)
        >>> executor.execute(pipeline, initial_state={}, context=ctx)
    """
    pipeline_name = "TradeOffPlot"

    pipeline: AnalysisPipeline[plt.Figure] = Pipeline[
        AnalysisPipelineState[plt.Figure], AnalysisPipelineContext
    ](name=pipeline_name)

    # Step 1: Load experiment predictions
    pipeline.add_step(
        name="LoadExperimentResults",
        task=load_experiment_results,
    )

    # Step 2: Compute evaluation metrics (kept as LazyFrame)
    pipeline.add_step(
        name="ComputeMetrics",
        task=compute_metrics,
    )

    # Step 3: Generate the trade-off plot
    pipeline.add_step(
        name="GenerateTradeOffPlot",
        task=generate_tradeoff_plot,  # type: ignore[arg-type]
    )

    return pipeline


__all__ = [
    "build_summary_table_pipeline",
    "build_tradeoff_plot_pipeline",
]
