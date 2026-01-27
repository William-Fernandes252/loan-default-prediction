"""Tasks package for analysis pipeline.

Provides reusable task components for analysis pipelines including
data loading, metrics computation, result generation, and persistence.
"""

from experiments.pipelines.analysis.tasks.common import (
    ComputeMetricsTask,
    ComputePerSeedMetricsTask,
    LoadExperimentResultsTask,
    compute_metrics,
    compute_per_seed_metrics,
    load_experiment_results,
)
from experiments.pipelines.analysis.tasks.generation import (
    GenerateCsVsResamplingPlotTask,
    GenerateImbalanceImpactPlotTask,
    GenerateMetricsHeatmapTask,
    GenerateStabilityPlotTask,
    GenerateSummaryTableTask,
    GenerateTradeOffPlotTask,
    create_imbalance_impact_task,
    create_stability_plot_task,
    create_summary_table_task,
    generate_cs_vs_resampling_plot,
    generate_imbalance_impact_plot,
    generate_metrics_heatmap,
    generate_stability_plot,
    generate_summary_table,
    generate_tradeoff_plot,
)
from experiments.pipelines.analysis.tasks.persistence import (
    SaveFigureArtifactTask,
    SaveTableArtifactTask,
    create_save_figure_task,
    create_save_table_task,
    save_figure_artifact,
    save_table_artifact,
)

__all__ = [
    # Common tasks
    "LoadExperimentResultsTask",
    "ComputeMetricsTask",
    "ComputePerSeedMetricsTask",
    "load_experiment_results",
    "compute_metrics",
    "compute_per_seed_metrics",
    # Generation tasks
    "GenerateSummaryTableTask",
    "GenerateTradeOffPlotTask",
    "GenerateStabilityPlotTask",
    "GenerateImbalanceImpactPlotTask",
    "GenerateCsVsResamplingPlotTask",
    "GenerateMetricsHeatmapTask",
    "create_summary_table_task",
    "create_stability_plot_task",
    "create_imbalance_impact_task",
    "generate_summary_table",
    "generate_tradeoff_plot",
    "generate_stability_plot",
    "generate_imbalance_impact_plot",
    "generate_cs_vs_resampling_plot",
    "generate_metrics_heatmap",
    # Persistence tasks
    "SaveTableArtifactTask",
    "SaveFigureArtifactTask",
    "create_save_table_task",
    "create_save_figure_task",
    "save_table_artifact",
    "save_figure_artifact",
]
