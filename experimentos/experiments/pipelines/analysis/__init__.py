"""Analysis pipelines package.

Provides pipeline components for analyzing experiment results including
factories, tasks, and pipeline definitions.
"""

from experiments.pipelines.analysis.base import (
    AnalysisPipelineFactory,
    ArtifactGenerator,
)
from experiments.pipelines.analysis.cost_sensitive_vs_resampling import (
    CostSensitiveVsResamplingComparisonPipelineFactory,
)
from experiments.pipelines.analysis.imbalance_impact import ImbalanceImpactAnalysisPipelineFactory
from experiments.pipelines.analysis.metrics_heatmap import MetricsHeatmapPipelineFactory
from experiments.pipelines.analysis.pipeline import (
    AnalysisArtifactRepository,
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineState,
    AnalysisPipelineTask,
    AnalysisPipelineTaskResult,
)
from experiments.pipelines.analysis.stability import StabilityAnalysisPipelineFactory
from experiments.pipelines.analysis.summary_table import SummaryTablePipelineFactory
from experiments.pipelines.analysis.tradeoff_plot import TradeoffPlotPipelineFactory

__all__ = [
    # Pipeline types
    "AnalysisPipeline",
    "AnalysisPipelineContext",
    "AnalysisPipelineState",
    "AnalysisPipelineTask",
    "AnalysisPipelineTaskResult",
    "AnalysisArtifactRepository",
    # Base factory
    "AnalysisPipelineFactory",
    "ArtifactGenerator",
    # Pipeline factory implementations
    "CostSensitiveVsResamplingComparisonPipelineFactory",
    "ImbalanceImpactAnalysisPipelineFactory",
    "MetricsHeatmapPipelineFactory",
    "StabilityAnalysisPipelineFactory",
    "SummaryTablePipelineFactory",
    "TradeoffPlotPipelineFactory",
]
