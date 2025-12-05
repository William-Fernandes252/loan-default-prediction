"""Analysis pipeline orchestrator.

This module provides the AnalysisPipeline class that coordinates
data loading, transformation, and export in a clean, composable way.
"""

from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from loguru import logger

from experiments.core.analysis.exporters import BaseExporter
from experiments.core.analysis.loaders import ParquetResultsLoader, ResultsPathProvider
from experiments.core.analysis.protocols import TranslationFunc
from experiments.core.analysis.transformers import BaseTransformer
from experiments.core.data import Dataset


class AnalysisType(Enum):
    """Supported analysis types."""

    STABILITY = "stability"
    RISK_TRADEOFF = "risk_tradeoff"
    IMBALANCE_IMPACT = "imbalance_impact"
    CS_VS_RESAMPLING = "cs_vs_resampling"
    HYPERPARAMETER = "hyperparameter"
    EXPERIMENT_SUMMARY = "experiment"


class OutputPathProvider(Protocol):
    """Protocol for providing output paths for analysis results."""

    def get_output_dir(
        self,
        dataset_id: str,
        is_figure: bool = True,
    ) -> Path:
        """Get the output directory for a dataset.

        Args:
            dataset_id: The dataset identifier.
            is_figure: Whether the output is a figure (affects path).

        Returns:
            The path to the output directory.
        """
        ...


class AnalysisPipeline:
    """Orchestrates the analysis pipeline: load → transform → export.

    This class coordinates the three stages of analysis using dependency
    injection for maximum flexibility and testability.

    Example:
        ```python
        loader = ParquetResultsLoader(ctx, translate)
        transformer = StabilityTransformer(translate)
        exporter = StabilityFigureExporter(translate)

        pipeline = AnalysisPipeline(
            loader=loader,
            transformer=transformer,
            exporter=exporter,
            output_path_provider=output_provider,
        )

        pipeline.run(Dataset.TAIWAN_CREDIT)
        ```
    """

    def __init__(
        self,
        loader: ParquetResultsLoader,
        transformer: BaseTransformer,
        exporter: BaseExporter,
        output_path_provider: OutputPathProvider,
        *,
        is_figure_output: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            loader: Data loader instance.
            transformer: Data transformer instance.
            exporter: Data exporter instance.
            output_path_provider: Provider for output directory paths.
            is_figure_output: Whether output is figures (affects path).
        """
        self._loader = loader
        self._transformer = transformer
        self._exporter = exporter
        self._output_path_provider = output_path_provider
        self._is_figure_output = is_figure_output

    def run(self, dataset: Dataset) -> list[Path]:
        """Execute the pipeline for a single dataset.

        Args:
            dataset: The dataset to analyze.

        Returns:
            List of paths to exported files, or empty list if no data.
        """
        # Load
        df = self._loader.load(dataset)
        if df.empty:
            logger.warning(f"No data found for {dataset.display_name}")
            return []

        # Transform
        transformed_data = self._transformer.transform(df, dataset)

        # Get output directory
        output_dir = self._output_path_provider.get_output_dir(
            dataset.id,
            is_figure=self._is_figure_output,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export
        exported_paths = self._exporter.export(transformed_data, output_dir, dataset)

        for path in exported_paths:
            logger.success(f"Exported: {path}")

        return exported_paths

    def run_all(self, datasets: list[Dataset] | None = None) -> dict[str, list[Path]]:
        """Execute the pipeline for multiple datasets.

        Args:
            datasets: List of datasets to analyze. If None, uses all datasets.

        Returns:
            Dictionary mapping dataset IDs to their exported file paths.
        """
        if datasets is None:
            datasets = list(Dataset)

        results: dict[str, list[Path]] = {}

        for dataset in datasets:
            logger.info(f"Running analysis for {dataset.display_name}...")
            results[dataset.id] = self.run(dataset)

        return results


class AnalysisPipelineFactory:
    """Factory for creating analysis pipelines using a registry pattern.

    This factory uses a registry pattern to create pipelines for different
    analysis types, making it easy to extend without modifying the class.
    """

    def __init__(
        self,
        path_provider: "ResultsPathProvider",
        output_path_provider: OutputPathProvider,
        translate: TranslationFunc,
    ) -> None:
        """Initialize the factory.

        Args:
            path_provider: Provider for results file paths.
            output_path_provider: Provider for output directory paths.
            translate: Translation function.
        """
        self._path_provider = path_provider
        self._output_path_provider = output_path_provider
        self._translate = translate
        self._registry: dict[AnalysisType, Callable[[], AnalysisPipeline]] = {
            AnalysisType.STABILITY: self._create_stability,
            AnalysisType.RISK_TRADEOFF: self._create_risk_tradeoff,
            AnalysisType.IMBALANCE_IMPACT: self._create_imbalance_impact,
            AnalysisType.CS_VS_RESAMPLING: self._create_cost_sensitive_vs_resampling,
            AnalysisType.HYPERPARAMETER: self._create_hyperparameter,
            AnalysisType.EXPERIMENT_SUMMARY: self._create_experiment_summary,
        }

    def _create_loader(self) -> ParquetResultsLoader:
        """Create a configured data loader."""
        return ParquetResultsLoader(self._path_provider, self._translate)

    def create(self, analysis_type: AnalysisType) -> AnalysisPipeline:
        """Create a pipeline for the given analysis type.

        Args:
            analysis_type: The type of analysis pipeline to create.

        Returns:
            A configured AnalysisPipeline instance.

        Raises:
            ValueError: If the analysis type is not registered.
        """
        if analysis_type not in self._registry:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        return self._registry[analysis_type]()

    def register(
        self,
        analysis_type: AnalysisType,
        factory_fn: Callable[[], AnalysisPipeline],
    ) -> None:
        """Register a custom pipeline factory.

        Args:
            analysis_type: The analysis type to register.
            factory_fn: A callable that returns an AnalysisPipeline instance.
        """
        self._registry[analysis_type] = factory_fn

    def create_stability_pipeline(self) -> AnalysisPipeline:
        """Create a pipeline for stability analysis.

        Deprecated: Use create(AnalysisType.STABILITY) instead.
        """
        return self._create_stability()

    def _create_stability(self) -> AnalysisPipeline:
        """Create a pipeline for stability analysis."""
        from experiments.core.analysis.exporters import StabilityFigureExporter
        from experiments.core.analysis.transformers import StabilityTransformer

        return AnalysisPipeline(
            loader=self._create_loader(),
            transformer=StabilityTransformer(self._translate),
            exporter=StabilityFigureExporter(self._translate),
            output_path_provider=self._output_path_provider,
        )

    def create_risk_tradeoff_pipeline(self) -> AnalysisPipeline:
        """Create a pipeline for risk tradeoff analysis.

        Deprecated: Use create(AnalysisType.RISK_TRADEOFF) instead.
        """
        return self._create_risk_tradeoff()

    def _create_risk_tradeoff(self) -> AnalysisPipeline:
        """Create a pipeline for risk tradeoff analysis."""
        from experiments.core.analysis.exporters import RiskTradeoffFigureExporter
        from experiments.core.analysis.transformers import RiskTradeoffTransformer

        return AnalysisPipeline(
            loader=self._create_loader(),
            transformer=RiskTradeoffTransformer(self._translate),
            exporter=RiskTradeoffFigureExporter(self._translate),
            output_path_provider=self._output_path_provider,
        )

    def create_imbalance_impact_pipeline(self) -> AnalysisPipeline:
        """Create a pipeline for imbalance impact analysis.

        Deprecated: Use create(AnalysisType.IMBALANCE_IMPACT) instead.
        """
        return self._create_imbalance_impact()

    def _create_imbalance_impact(self) -> AnalysisPipeline:
        """Create a pipeline for imbalance impact analysis."""
        from experiments.core.analysis.exporters import ImbalanceImpactFigureExporter
        from experiments.core.analysis.transformers import ImbalanceImpactTransformer

        return AnalysisPipeline(
            loader=self._create_loader(),
            transformer=ImbalanceImpactTransformer(self._translate),
            exporter=ImbalanceImpactFigureExporter(self._translate),
            output_path_provider=self._output_path_provider,
        )

    def create_cost_sensitive_vs_resampling_pipeline(self) -> AnalysisPipeline:
        """Create a pipeline for cost-sensitive vs resampling analysis.

        Deprecated: Use create(AnalysisType.CS_VS_RESAMPLING) instead.
        """
        return self._create_cost_sensitive_vs_resampling()

    def _create_cost_sensitive_vs_resampling(self) -> AnalysisPipeline:
        """Create a pipeline for cost-sensitive vs resampling analysis."""
        from experiments.core.analysis.exporters import (
            CostSensitiveVsResamplingFigureExporter,
        )
        from experiments.core.analysis.transformers import (
            CostSensitiveVsResamplingTransformer,
        )

        return AnalysisPipeline(
            loader=self._create_loader(),
            transformer=CostSensitiveVsResamplingTransformer(self._translate),
            exporter=CostSensitiveVsResamplingFigureExporter(self._translate),
            output_path_provider=self._output_path_provider,
        )

    def create_hyperparameter_pipeline(self) -> AnalysisPipeline:
        """Create a pipeline for hyperparameter effects analysis.

        Deprecated: Use create(AnalysisType.HYPERPARAMETER) instead.
        """
        return self._create_hyperparameter()

    def _create_hyperparameter(self) -> AnalysisPipeline:
        """Create a pipeline for hyperparameter effects analysis."""
        from experiments.core.analysis.exporters import HyperparameterFigureExporter
        from experiments.core.analysis.transformers import HyperparameterTransformer

        return AnalysisPipeline(
            loader=self._create_loader(),
            transformer=HyperparameterTransformer(self._translate),
            exporter=HyperparameterFigureExporter(self._translate),
            output_path_provider=self._output_path_provider,
        )

    def create_experiment_summary_pipeline(self) -> AnalysisPipeline:
        """Create a pipeline for experiment summary analysis.

        This pipeline exports to multiple formats: PNG tables, CSV, and LaTeX.

        Deprecated: Use create(AnalysisType.EXPERIMENT_SUMMARY) instead.
        """
        return self._create_experiment_summary()

    def _create_experiment_summary(self) -> AnalysisPipeline:
        """Create a pipeline for experiment summary analysis.

        This pipeline exports to multiple formats: PNG tables, CSV, and LaTeX.
        """
        from experiments.core.analysis.exporters import (
            CompositeExporter,
            CsvExporter,
            ExperimentSummaryFigureExporter,
            LatexExporter,
        )
        from experiments.core.analysis.transformers import ExperimentSummaryTransformer

        # Composite exporter for multiple output formats
        exporter = CompositeExporter(
            self._translate,
            exporters=[
                ExperimentSummaryFigureExporter(self._translate),
                CsvExporter(self._translate),
                LatexExporter(self._translate),
            ],
        )

        return AnalysisPipeline(
            loader=self._create_loader(),
            transformer=ExperimentSummaryTransformer(self._translate),
            exporter=exporter,
            output_path_provider=self._output_path_provider,
            is_figure_output=False,  # Summary goes to base dir, not figures
        )


__all__ = [
    "AnalysisPipeline",
    "AnalysisPipelineFactory",
    "AnalysisType",
    "OutputPathProvider",
]
