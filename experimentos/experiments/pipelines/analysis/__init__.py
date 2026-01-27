"""Analysis pipelines package.

Provides pipeline components for analyzing experiment results including
factories, tasks, and pipeline definitions.
"""

from experiments.pipelines.analysis.factory import (
    build_summary_table_pipeline,
    build_tradeoff_plot_pipeline,
)
from experiments.pipelines.analysis.pipeline import (
    AnalysisArtifactRepository,
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineState,
    AnalysisPipelineTask,
    AnalysisPipelineTaskResult,
)

__all__ = [
    # Pipeline types
    "AnalysisPipeline",
    "AnalysisPipelineContext",
    "AnalysisPipelineState",
    "AnalysisPipelineTask",
    "AnalysisPipelineTaskResult",
    "AnalysisArtifactRepository",
    # Factory functions
    "build_summary_table_pipeline",
    "build_tradeoff_plot_pipeline",
]
