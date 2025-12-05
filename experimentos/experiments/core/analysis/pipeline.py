"""Analysis pipeline orchestrator.

This module provides the AnalysisPipeline class that coordinates
data loading, transformation, and export in a clean, composable way.
"""

from pathlib import Path
from typing import Protocol

from loguru import logger

from experiments.core.analysis.exporters import BaseExporter
from experiments.core.analysis.loaders import (
    EnrichedResultsLoader,
    ResultsPathProvider,
)
from experiments.core.analysis.protocols import DataLoader, TranslationFunc
from experiments.core.analysis.transformers import BaseTransformer
from experiments.core.data import Dataset


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
        loader = EnrichedResultsLoader(ctx, translate)
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
        loader: "DataLoader",
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
    """Factory for creating pre-configured analysis pipelines.

    This factory simplifies creating pipelines for common analysis types
    by encapsulating the component wiring.
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

    def _create_loader(self) -> DataLoader:
        """Create a configured data loader."""
        return EnrichedResultsLoader(self._path_provider, self._translate)

    def create_stability_pipeline(self) -> AnalysisPipeline:
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
    "OutputPathProvider",
]
