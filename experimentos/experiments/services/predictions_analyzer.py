"""Service for analyzing model predictions and generating analysis artifacts.

This module provides the PredictionsAnalyzer service which centralizes all
analysis logic, including pipeline creation, context building, and artifact
generation. It encapsulates the complexity of running different analysis types.
"""

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from pydantic import BaseModel

from experiments.config.settings import LdpSettings
from experiments.core.analysis import Locale
from experiments.core.analysis.evaluation import ModelResultsEvaluator
from experiments.core.analysis.metrics import Metric
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import Technique
from experiments.core.predictions.repository import ModelPredictionsRepository
from experiments.lib.pipelines.execution import PipelineExecutor
from experiments.pipelines.analysis.base import AnalysisArtifactRepository, ArtifactGenerator
from experiments.pipelines.analysis.cost_sensitive_vs_resampling import (
    CostSensitiveVsResamplingComparisonPipelineFactory,
)
from experiments.pipelines.analysis.imbalance_impact import ImbalanceImpactAnalysisPipelineFactory
from experiments.pipelines.analysis.metrics_heatmap import MetricsHeatmapPipelineFactory
from experiments.pipelines.analysis.pipeline import AnalysisPipelineContext
from experiments.pipelines.analysis.stability import StabilityAnalysisPipelineFactory
from experiments.pipelines.analysis.summary_table import SummaryTablePipelineFactory
from experiments.pipelines.analysis.tradeoff_plot import TradeoffPlotPipelineFactory
from experiments.services.translator import create_translator


class AnalysisType(StrEnum):
    """Available analysis types."""

    SUMMARY_TABLE = "summary_table"
    TRADEOFF_PLOT = "tradeoff_plot"
    STABILITY_PLOT = "stability_plot"
    IMBALANCE_IMPACT_PLOT = "imbalance_impact_plot"
    CS_VS_RESAMPLING_PLOT = "cs_vs_resampling_plot"
    METRICS_HEATMAP = "metrics_heatmap"


class AnalysisParams(BaseModel):
    """Parameters for running an analysis.

    Attributes:
        dataset: The dataset to analyze.
        locale: Locale for internationalization.
        force_overwrite: Whether to overwrite existing artifacts.
        use_gpu: Whether to use GPU acceleration.
        execution_id: Specific execution ID to analyze. If None, uses the latest.
        metric: Metric to visualize (for stability and imbalance impact plots).
        technique_filter: Filter results by a specific technique (for summary table).
    """

    dataset: Dataset
    locale: Locale = Locale.EN_US
    force_overwrite: bool = False
    use_gpu: bool = False
    execution_id: str | None = None
    metric: Metric = Metric.BALANCED_ACCURACY
    technique_filter: Technique | None = None

    model_config = {"frozen": True}


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Result of an analysis execution.

    Attributes:
        dataset: The dataset that was analyzed.
        analysis_type: The type of analysis performed.
        success: Whether the analysis succeeded.
        error_message: Error message if the analysis failed.
    """

    dataset: Dataset
    analysis_type: AnalysisType
    success: bool
    error_message: str | None = None

    def succeeded(self) -> bool:
        """Check if the analysis succeeded."""
        return self.success


class PredictionsAnalyzer:
    """Service for analyzing model predictions and generating artifacts.

    This service encapsulates all analysis logic, including:
    - Pipeline factory instantiation based on analysis type
    - Context creation with injected dependencies
    - Artifact generator selection
    - Pipeline execution

    The CLI layer can use this service to delegate all analysis work,
    keeping the CLI commands simple and focused on argument parsing.
    """

    def __init__(
        self,
        analysis_artifacts_repository: AnalysisArtifactRepository,
        predictions_repository: ModelPredictionsRepository,
        results_evaluator: ModelResultsEvaluator,
        settings: LdpSettings,
    ) -> None:
        """Initialize the predictions analyzer.

        Args:
            analysis_artifacts_repository: Repository for saving analysis artifacts.
            predictions_repository: Repository for loading model predictions.
            results_evaluator: Evaluator for computing analysis metrics.
            settings: Application settings.
        """
        self._analysis_artifacts_repository = analysis_artifacts_repository
        self._predictions_repository = predictions_repository
        self._results_evaluator = results_evaluator
        self._settings = settings

    def run_analysis(
        self,
        analysis_type: AnalysisType,
        params: AnalysisParams,
    ) -> AnalysisResult:
        """Run a single analysis for a dataset.

        Args:
            analysis_type: The type of analysis to run.
            params: Parameters for the analysis.

        Returns:
            AnalysisResult indicating success or failure.
        """
        # Determine analysis name (artifact filename)
        analysis_name = self._get_analysis_name(analysis_type, params)

        # Create the pipeline factory
        factory = self._create_factory(analysis_type, params)

        # Get the appropriate artifact generator
        artifact_generator = self._get_artifact_generator(analysis_type)

        # Build the pipeline
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name(self._get_pipeline_params(analysis_type, params)),
            artifact_generator=artifact_generator,  # type: ignore[arg-type]
        )

        # Create the context
        context = self._create_context(params, analysis_name)

        # Create a fresh executor for each analysis (executors are not reusable)
        executor = PipelineExecutor(max_workers=1)

        # Execute the pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            return AnalysisResult(
                dataset=params.dataset,
                analysis_type=analysis_type,
                success=True,
            )
        else:
            error = result.last_error()
            return AnalysisResult(
                dataset=params.dataset,
                analysis_type=analysis_type,
                success=False,
                error_message=str(error) if error else "Unknown error",
            )

    def _get_analysis_name(self, analysis_type: AnalysisType, params: AnalysisParams) -> str:
        """Get the artifact filename for an analysis type.

        Args:
            analysis_type: The type of analysis.
            params: The analysis parameters.

        Returns:
            The artifact filename with extension.
        """
        match analysis_type:
            case AnalysisType.SUMMARY_TABLE:
                if params.technique_filter:
                    return f"summary_table_{params.technique_filter.value}.tex"
                return "summary_table.tex"
            case AnalysisType.TRADEOFF_PLOT:
                return "tradeoff_plot.png"
            case AnalysisType.STABILITY_PLOT:
                return f"stability_plot_{params.metric.value}.png"
            case AnalysisType.IMBALANCE_IMPACT_PLOT:
                return f"imbalance_impact_{params.metric.value}.png"
            case AnalysisType.CS_VS_RESAMPLING_PLOT:
                return "cs_vs_resampling_plot.png"
            case AnalysisType.METRICS_HEATMAP:
                return "metrics_heatmap.png"

    def _create_factory(self, analysis_type: AnalysisType, params: AnalysisParams) -> Any:
        """Create the appropriate pipeline factory for the analysis type.

        Args:
            analysis_type: The type of analysis.
            params: The analysis parameters.

        Returns:
            The configured pipeline factory.
        """
        match analysis_type:
            case AnalysisType.SUMMARY_TABLE:
                return SummaryTablePipelineFactory(
                    analysis_artifacts_repository=self._analysis_artifacts_repository,
                    technique_filter=params.technique_filter,
                )
            case AnalysisType.TRADEOFF_PLOT:
                return TradeoffPlotPipelineFactory(
                    analysis_artifacts_repository=self._analysis_artifacts_repository,
                )
            case AnalysisType.STABILITY_PLOT:
                return StabilityAnalysisPipelineFactory(
                    analysis_artifacts_repository=self._analysis_artifacts_repository,
                    metric=params.metric,
                )
            case AnalysisType.IMBALANCE_IMPACT_PLOT:
                return ImbalanceImpactAnalysisPipelineFactory(
                    analysis_artifacts_repository=self._analysis_artifacts_repository,
                    metric=params.metric,
                )
            case AnalysisType.CS_VS_RESAMPLING_PLOT:
                return CostSensitiveVsResamplingComparisonPipelineFactory(
                    analysis_artifacts_repository=self._analysis_artifacts_repository,
                )
            case AnalysisType.METRICS_HEATMAP:
                return MetricsHeatmapPipelineFactory(
                    analysis_artifacts_repository=self._analysis_artifacts_repository,
                )

    def _get_artifact_generator(self, analysis_type: AnalysisType) -> ArtifactGenerator:
        """Get the appropriate artifact generator for the analysis type.

        Args:
            analysis_type: The type of analysis.

        Returns:
            The artifact generator function.
        """
        if analysis_type == AnalysisType.SUMMARY_TABLE:
            return self._generate_summary_table_artifact
        return self._generate_figure_artifact

    def _get_pipeline_params(
        self, analysis_type: AnalysisType, params: AnalysisParams
    ) -> dict[str, str]:
        """Get the parameters for pipeline naming.

        Args:
            analysis_type: The type of analysis.
            params: The analysis parameters.

        Returns:
            Dictionary of pipeline parameters for naming.
        """
        base_params = {"dataset": params.dataset.value}

        match analysis_type:
            case AnalysisType.STABILITY_PLOT | AnalysisType.IMBALANCE_IMPACT_PLOT:
                return {**base_params, "metric": params.metric.value}
            case _:
                return base_params

    def _create_context(
        self, params: AnalysisParams, analysis_name: str
    ) -> AnalysisPipelineContext:
        """Create the analysis pipeline context.

        Args:
            params: The analysis parameters.
            analysis_name: The artifact filename.

        Returns:
            Configured AnalysisPipelineContext.
        """
        # Resolve locale: params > settings default
        resolved_locale = params.locale or Locale(self._settings.locale)

        # Create translator
        translator = create_translator(resolved_locale)

        return AnalysisPipelineContext(
            dataset=params.dataset,
            analysis_name=analysis_name,
            predictions_repository=self._predictions_repository,
            results_evaluator=self._results_evaluator,
            analysis_artifacts_repository=self._analysis_artifacts_repository,
            use_gpu=params.use_gpu,
            force_overwrite=params.force_overwrite,
            locale=resolved_locale,
            translator=translator,
            execution_id=params.execution_id,
        )

    @staticmethod
    def _generate_summary_table_artifact(
        result_data: pl.DataFrame, context: AnalysisPipelineContext
    ) -> BytesIO:
        """Generate a LaTeX table artifact from a DataFrame.

        Args:
            result_data: The summary table DataFrame.
            context: The analysis pipeline context.

        Returns:
            BytesIO containing the LaTeX table bytes.
        """
        # Convert Polars DataFrame to pandas for to_latex() support
        pdf = result_data.to_pandas()

        # Generate LaTeX table string
        latex_str = pdf.to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
            caption=f"Results for {context.dataset.value}",
            label=f"tab:{context.analysis_name}",
        )

        # Convert to bytes
        artifact_bytes = latex_str.encode("utf-8")
        return BytesIO(artifact_bytes)

    @staticmethod
    def _generate_figure_artifact(
        result_data: plt.Figure, context: AnalysisPipelineContext
    ) -> BytesIO:
        """Generate a PNG figure artifact from a matplotlib Figure.

        Args:
            result_data: The matplotlib Figure.
            context: The analysis pipeline context.

        Returns:
            BytesIO containing the PNG figure bytes.
        """
        # Render figure to bytes
        buffer = BytesIO()
        result_data.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        # Close the figure to free memory
        plt.close(result_data)

        return buffer
