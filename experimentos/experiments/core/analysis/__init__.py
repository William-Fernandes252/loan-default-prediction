"""Analysis module for experimental results.

This module provides a pipeline-based architecture for analyzing
experimental results using dependency injection and the dependency
inversion principle.
"""

from experiments.core.analysis.exporters import (
    BaseExporter,
    CompositeExporter,
    CostSensitiveVsResamplingFigureExporter,
    CsvExporter,
    ExperimentSummaryFigureExporter,
    FigureExporter,
    HyperparameterFigureExporter,
    ImbalanceImpactFigureExporter,
    LatexExporter,
    RiskTradeoffFigureExporter,
    StabilityFigureExporter,
)
from experiments.core.analysis.loaders import (
    DisplayColumnEnricher,
    EnrichedResultsLoader,
    ParquetResultsLoader,
    ResultsPathProvider,
)
from experiments.core.analysis.metrics import (
    IMBALANCE_RATIOS,
    MetricConfig,
    get_metric_configs,
)
from experiments.core.analysis.pipeline import (
    AnalysisPipeline,
    AnalysisPipelineFactory,
    AnalysisType,
    OutputPathProvider,
)
from experiments.core.analysis.protocols import (
    DataExporter,
    DataLoader,
    DataTransformer,
    TranslationFunc,
)
from experiments.core.analysis.transformers import (
    BaseTransformer,
    CostSensitiveVsResamplingTransformer,
    ExperimentSummaryTransformer,
    HyperparameterTransformer,
    ImbalanceImpactTransformer,
    RiskTradeoffTransformer,
    StabilityTransformer,
)

__all__ = [
    # Protocols
    "DataLoader",
    "DataTransformer",
    "DataExporter",
    "TranslationFunc",
    "ResultsPathProvider",
    "OutputPathProvider",
    # Constants
    "IMBALANCE_RATIOS",
    "MetricConfig",
    "get_metric_configs",
    "get_metric_display_names",
    "translate_metric",
    # Loaders
    "ParquetResultsLoader",
    "DisplayColumnEnricher",
    "EnrichedResultsLoader",
    # Transformers
    "BaseTransformer",
    "StabilityTransformer",
    "RiskTradeoffTransformer",
    "ImbalanceImpactTransformer",
    "CostSensitiveVsResamplingTransformer",
    "HyperparameterTransformer",
    "ExperimentSummaryTransformer",
    # Exporters
    "BaseExporter",
    "FigureExporter",
    "StabilityFigureExporter",
    "RiskTradeoffFigureExporter",
    "ImbalanceImpactFigureExporter",
    "CostSensitiveVsResamplingFigureExporter",
    "HyperparameterFigureExporter",
    "ExperimentSummaryFigureExporter",
    "CsvExporter",
    "LatexExporter",
    "CompositeExporter",
    # Pipeline
    "AnalysisPipeline",
    "AnalysisPipelineFactory",
    "AnalysisType",
]
