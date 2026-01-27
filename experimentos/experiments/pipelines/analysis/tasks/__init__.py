"""Tasks package for analysis pipeline.

Provides reusable task components for analysis pipelines including
data loading, metrics computation, result generation, and persistence.
"""

from experiments.pipelines.analysis.tasks.common import (
    ComputeMetricsTask,
    LoadExperimentResultsTask,
    compute_metrics,
    load_experiment_results,
)
from experiments.pipelines.analysis.tasks.generation import (
    GenerateSummaryTableTask,
    GenerateTradeOffPlotTask,
    create_summary_table_task,
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
    "load_experiment_results",
    "compute_metrics",
    # Generation tasks
    "GenerateSummaryTableTask",
    "GenerateTradeOffPlotTask",
    "create_summary_table_task",
    "generate_summary_table",
    "generate_tradeoff_plot",
    # Persistence tasks
    "SaveTableArtifactTask",
    "SaveFigureArtifactTask",
    "create_save_table_task",
    "create_save_figure_task",
    "save_table_artifact",
    "save_figure_artifact",
]
