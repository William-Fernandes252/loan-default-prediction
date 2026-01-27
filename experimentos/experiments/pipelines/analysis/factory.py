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

from experiments.core.analysis.metrics import Metric
from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines import Pipeline
from experiments.pipelines.analysis.pipeline import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineState,
)
from experiments.pipelines.analysis.tasks.common import (
    compute_metrics,
    compute_per_seed_metrics,
    load_experiment_results,
)
from experiments.pipelines.analysis.tasks.generation import (
    create_imbalance_impact_task,
    create_stability_plot_task,
    create_summary_table_task,
    generate_cs_vs_resampling_plot,
    generate_metrics_heatmap,
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


def build_stability_pipeline(
    metric: Metric = Metric.BALANCED_ACCURACY,
) -> AnalysisPipeline[plt.Figure]:
    """Build a pipeline for generating stability boxplots.

    Creates a pipeline with the following steps:
    1. LoadExperimentResults: Fetches predictions from repository
    2. ComputePerSeedMetrics: Computes per-seed evaluation metrics (lazy)
    3. GenerateStabilityPlot: Creates the boxplot visualization

    Persistence steps should be appended by the caller using `pipeline.add_step()`.

    Args:
        metric: The metric to visualize in the boxplot. Defaults to balanced accuracy.

    Returns:
        AnalysisPipeline[plt.Figure]: A configured pipeline that produces
        a matplotlib Figure in the `result_data` state key.
    """
    pipeline_name = f"StabilityPlot_{metric.value}"

    pipeline: AnalysisPipeline[plt.Figure] = Pipeline[
        AnalysisPipelineState[plt.Figure], AnalysisPipelineContext
    ](name=pipeline_name)

    # Step 1: Load experiment predictions
    pipeline.add_step(
        name="LoadExperimentResults",
        task=load_experiment_results,
    )

    # Step 2: Compute per-seed metrics (kept as LazyFrame)
    pipeline.add_step(
        name="ComputePerSeedMetrics",
        task=compute_per_seed_metrics,
    )

    # Step 3: Generate the stability plot
    pipeline.add_step(
        name="GenerateStabilityPlot",
        task=create_stability_plot_task(metric=metric),  # type: ignore[arg-type]
    )

    return pipeline


def build_imbalance_impact_pipeline(
    metric: Metric = Metric.BALANCED_ACCURACY,
) -> AnalysisPipeline[plt.Figure]:
    """Build a pipeline for generating imbalance impact scatter plots.

    Creates a pipeline with the following steps:
    1. LoadExperimentResults: Fetches predictions from repository
    2. ComputeMetrics: Computes aggregated evaluation metrics (lazy)
    3. GenerateImbalanceImpactPlot: Creates the scatter plot visualization

    Persistence steps should be appended by the caller using `pipeline.add_step()`.

    Args:
        metric: The metric to visualize. Defaults to balanced accuracy.

    Returns:
        AnalysisPipeline[plt.Figure]: A configured pipeline that produces
        a matplotlib Figure in the `result_data` state key.
    """
    pipeline_name = f"ImbalanceImpactPlot_{metric.value}"

    pipeline: AnalysisPipeline[plt.Figure] = Pipeline[
        AnalysisPipelineState[plt.Figure], AnalysisPipelineContext
    ](name=pipeline_name)

    # Step 1: Load experiment predictions
    pipeline.add_step(
        name="LoadExperimentResults",
        task=load_experiment_results,
    )

    # Step 2: Compute aggregated metrics (kept as LazyFrame)
    pipeline.add_step(
        name="ComputeMetrics",
        task=compute_metrics,
    )

    # Step 3: Generate the imbalance impact plot
    pipeline.add_step(
        name="GenerateImbalanceImpactPlot",
        task=create_imbalance_impact_task(metric=metric),  # type: ignore[arg-type]
    )

    return pipeline


def build_cs_vs_resampling_pipeline() -> AnalysisPipeline[plt.Figure]:
    """Build a pipeline for comparing cost-sensitive vs resampling techniques.

    Creates a pipeline with the following steps:
    1. LoadExperimentResults: Fetches predictions from repository
    2. ComputePerSeedMetrics: Computes per-seed evaluation metrics (lazy)
    3. GenerateCsVsResamplingPlot: Creates the bar plot visualization

    Persistence steps should be appended by the caller using `pipeline.add_step()`.

    Returns:
        AnalysisPipeline[plt.Figure]: A configured pipeline that produces
        a matplotlib Figure in the `result_data` state key.
    """
    pipeline_name = "CsVsResamplingPlot"

    pipeline: AnalysisPipeline[plt.Figure] = Pipeline[
        AnalysisPipelineState[plt.Figure], AnalysisPipelineContext
    ](name=pipeline_name)

    # Step 1: Load experiment predictions
    pipeline.add_step(
        name="LoadExperimentResults",
        task=load_experiment_results,
    )

    # Step 2: Compute per-seed metrics (kept as LazyFrame)
    pipeline.add_step(
        name="ComputePerSeedMetrics",
        task=compute_per_seed_metrics,
    )

    # Step 3: Generate the comparison plot
    pipeline.add_step(
        name="GenerateCsVsResamplingPlot",
        task=generate_cs_vs_resampling_plot,  # type: ignore[arg-type]
    )

    return pipeline


def build_metrics_heatmap_pipeline() -> AnalysisPipeline[plt.Figure]:
    """Build a pipeline for generating metrics heatmaps.

    Creates a pipeline with the following steps:
    1. LoadExperimentResults: Fetches predictions from repository
    2. ComputeMetrics: Computes aggregated evaluation metrics (lazy)
    3. GenerateMetricsHeatmap: Creates the heatmap visualization

    Persistence steps should be appended by the caller using `pipeline.add_step()`.

    Returns:
        AnalysisPipeline[plt.Figure]: A configured pipeline that produces
        a matplotlib Figure in the `result_data` state key.
    """
    pipeline_name = "MetricsHeatmap"

    pipeline: AnalysisPipeline[plt.Figure] = Pipeline[
        AnalysisPipelineState[plt.Figure], AnalysisPipelineContext
    ](name=pipeline_name)

    # Step 1: Load experiment predictions
    pipeline.add_step(
        name="LoadExperimentResults",
        task=load_experiment_results,
    )

    # Step 2: Compute aggregated metrics (kept as LazyFrame)
    pipeline.add_step(
        name="ComputeMetrics",
        task=compute_metrics,
    )

    # Step 3: Generate the heatmap
    pipeline.add_step(
        name="GenerateMetricsHeatmap",
        task=generate_metrics_heatmap,  # type: ignore[arg-type]
    )

    return pipeline


__all__ = [
    "build_summary_table_pipeline",
    "build_tradeoff_plot_pipeline",
    "build_stability_pipeline",
    "build_imbalance_impact_pipeline",
    "build_cs_vs_resampling_pipeline",
    "build_metrics_heatmap_pipeline",
]
