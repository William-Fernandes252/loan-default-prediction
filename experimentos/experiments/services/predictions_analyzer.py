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
from experiments.core.modeling.classifiers import ModelType, Technique
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
from experiments.pipelines.analysis.tasks.display_labels import create_export_dataframe
from experiments.pipelines.analysis.tradeoff_plot import TradeoffPlotPipelineFactory
from experiments.services.analysis_artifacts_repository import AnalysisArtifactsRepository
from experiments.services.translator import GettextTranslator, create_translator


class AnalysisType(StrEnum):
    """Available analysis types."""

    SUMMARY_TABLE = "summary_table"
    TRADEOFF_PLOT = "tradeoff_plot"
    STABILITY_PLOT = "stability_plot"
    IMBALANCE_IMPACT_PLOT = "imbalance_impact_plot"
    CS_VS_RESAMPLING_PLOT = "cs_vs_resampling_plot"
    METRICS_HEATMAP = "metrics_heatmap"
    CROSS_DATASET_TABLE = "cross_dataset_table"


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
        pdf = create_export_dataframe(result_data.to_pandas(), context)

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

    # --- Cross-dataset analysis (not pipeline-based) ---

    _REAL_DATASETS = [
        Dataset.LENDING_CLUB,
        Dataset.TAIWAN_CREDIT,
        Dataset.CORPORATE_CREDIT_RATING,
    ]

    def run_cross_dataset_analysis(
        self,
        *,
        technique_filter: Technique | None = None,
        locale: Locale = Locale.EN_US,
        force_overwrite: bool = False,
        execution_id: str | None = None,
    ) -> list[AnalysisResult]:
        """Generate cross-dataset comparison tables for balanced accuracy.

        Produces a LaTeX table per technique (or for a single technique if
        filtered) where rows are model types and columns are datasets.
        Each cell contains the mean ± std of balanced accuracy.

        Args:
            technique_filter: If set, only generate a table for this technique.
            locale: Locale for internationalization.
            force_overwrite: Whether to overwrite existing artifacts.
            execution_id: Specific execution ID. If None, uses the latest.

        Returns:
            List of AnalysisResult, one per generated table.
        """
        translator = create_translator(locale)
        repo = self._analysis_artifacts_repository
        assert isinstance(repo, AnalysisArtifactsRepository)

        # 1. Gather metrics from every real dataset
        all_metrics: list[pl.LazyFrame] = []
        for ds in self._REAL_DATASETS:
            predictions = (
                self._predictions_repository.get_predictions_for_execution(ds, execution_id)
                if execution_id
                else self._predictions_repository.get_latest_predictions_for_experiment(ds)
            )
            if predictions is None:
                continue
            metrics_lf = self._results_evaluator.evaluate(predictions)
            all_metrics.append(metrics_lf.with_columns(pl.lit(ds.value).alias("dataset")))

        if not all_metrics:
            return [
                AnalysisResult(
                    dataset=Dataset.LENDING_CLUB,
                    analysis_type=AnalysisType.CROSS_DATASET_TABLE,
                    success=False,
                    error_message="No predictions found for any dataset.",
                )
            ]

        combined_df: pl.DataFrame = pl.concat(all_metrics).collect()

        # 2. Determine techniques to process
        available_techniques = combined_df["technique"].unique().to_list()
        if technique_filter is not None:
            techniques = [technique_filter.value]
        else:
            techniques = sorted(available_techniques)

        # 3. Build and save one table per technique
        results: list[AnalysisResult] = []
        for technique_val in techniques:
            artifact_name = f"cross_dataset_comparison_{technique_val}.tex"

            if not force_overwrite and repo.cross_dataset_artifact_exists(
                artifact_name, locale=locale.value
            ):
                results.append(
                    AnalysisResult(
                        dataset=Dataset.LENDING_CLUB,
                        analysis_type=AnalysisType.CROSS_DATASET_TABLE,
                        success=True,
                    )
                )
                continue

            table_df = self._build_cross_dataset_table(combined_df, technique_val, translator)

            # Generate LaTeX artifact
            pdf = table_df.to_pandas()
            technique_display = self._technique_display(technique_val, translator)
            latex_str = pdf.to_latex(
                index=False,
                escape=False,
                caption=f"Balanced Accuracy — {technique_display}",
                label=f"tab:cross_dataset_{technique_val}",
            )
            artifact_bytes = BytesIO(latex_str.encode("utf-8"))

            repo.save_cross_dataset_artifact(artifact_name, artifact_bytes, locale=locale.value)
            results.append(
                AnalysisResult(
                    dataset=Dataset.LENDING_CLUB,
                    analysis_type=AnalysisType.CROSS_DATASET_TABLE,
                    success=True,
                )
            )

        return results

    def _build_cross_dataset_table(
        self,
        combined_df: pl.DataFrame,
        technique: str,
        translator: GettextTranslator,
    ) -> pl.DataFrame:
        """Build a cross-dataset comparison table for one technique.

        Rows are model types, columns are datasets.  Each cell is formatted
        as ``mean ± std`` of balanced accuracy.

        Args:
            combined_df: Metrics from all datasets with a ``dataset`` column.
            technique: The technique value to filter by.
            translator: Translator for display names.

        Returns:
            A Polars DataFrame ready for LaTeX export.
        """
        ba_mean = f"{Metric.BALANCED_ACCURACY}_mean"
        ba_std = f"{Metric.BALANCED_ACCURACY}_std"

        filtered = combined_df.filter(pl.col("technique") == technique)

        # Format cells as "mean ± std"
        formatted = filtered.with_columns(
            (
                pl.col(ba_mean).round(4).cast(pl.String)
                + " ± "
                + pl.col(ba_std).round(4).cast(pl.String)
            ).alias("cell")
        )

        # Map display names
        dataset_display = {
            ds.value: self._dataset_display(ds, translator) for ds in self._REAL_DATASETS
        }
        model_display = {
            mt.value: self._model_type_display(mt.value, translator) for mt in ModelType
        }

        formatted = formatted.with_columns(
            pl.col("dataset").replace_strict(dataset_display).alias("dataset"),
            pl.col("model_type").replace_strict(model_display).alias("model_type"),
        )

        # Pivot: rows = model_type, columns = dataset
        pivoted = formatted.select("model_type", "dataset", "cell").pivot(
            on="dataset",
            index="model_type",
            values="cell",
        )

        # Reorder columns: model_type first, then datasets in canonical order
        ordered_ds_cols = [
            dataset_display[ds.value]
            for ds in self._REAL_DATASETS
            if dataset_display[ds.value] in pivoted.columns
        ]
        return pivoted.select(
            pl.col("model_type").alias(translator.translate("Model")),
            *[pl.col(c) for c in ordered_ds_cols],
        ).sort(translator.translate("Model"))

    # --- Display-name helpers (reuse existing label maps) ---

    @staticmethod
    def _dataset_display(ds: Dataset, translator: GettextTranslator) -> str:
        from experiments.pipelines.analysis.tasks.display_labels import (
            _DATASET_DISPLAY_NAMES,
        )

        name = _DATASET_DISPLAY_NAMES.get(ds, ds.value)
        return translator.translate(name)

    @staticmethod
    def _model_type_display(model_type: str, translator: GettextTranslator) -> str:
        from experiments.pipelines.analysis.tasks.display_labels import (
            _MODEL_TYPE_DISPLAY_NAMES,
        )

        name = _MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type.replace("_", " ").title())
        return translator.translate(name)

    @staticmethod
    def _technique_display(technique: str, translator: GettextTranslator) -> str:
        from experiments.pipelines.analysis.tasks.display_labels import (
            _TECHNIQUE_DISPLAY_NAMES,
        )

        name = _TECHNIQUE_DISPLAY_NAMES.get(technique, technique.replace("_", " ").title())
        return translator.translate(name)
