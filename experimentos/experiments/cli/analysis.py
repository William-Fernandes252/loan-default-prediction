"""CLI for analyzing experimental results.

This module provides commands for various analysis types on the resultant models
from the experiments. Each command generates visualizations and reports to help
understand model performance, stability, and the effects of different techniques.

The CLI uses the abstract factory pattern:
- Pipeline factories build complete pipelines with analysis steps and artifact generation
- Artifact generators are injected at runtime based on the analysis type
"""

import enum
from io import BytesIO
import sys
from typing import Optional

from loguru import logger
import matplotlib.pyplot as plt
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.containers import container
from experiments.core.analysis import Locale
from experiments.core.analysis.metrics import Metric
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines.execution import PipelineExecutor
from experiments.pipelines.analysis.cost_sensitive_vs_resampling import (
    CostSensitiveVsResamplingComparisonPipelineFactory,
)
from experiments.pipelines.analysis.imbalance_impact import ImbalanceImpactAnalysisPipelineFactory
from experiments.pipelines.analysis.metrics_heatmap import MetricsHeatmapPipelineFactory
from experiments.pipelines.analysis.pipeline import AnalysisPipelineContext
from experiments.pipelines.analysis.stability import StabilityAnalysisPipelineFactory
from experiments.pipelines.analysis.summary_table import SummaryTablePipelineFactory
from experiments.pipelines.analysis.tradeoff_plot import TradeoffPlotPipelineFactory
from experiments.services.analysis_artifacts_repository import AnalysisArtifactsRepository
from experiments.services.model_predictions_repository import ModelPredictionsStorageRepository
from experiments.services.model_results_evaluator import ModelResultsEvaluatorImpl
from experiments.services.translator import create_translator


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


class AnalysisType(enum.StrEnum):
    """Available analysis types."""

    SUMMARY_TABLE = "summary_table"
    TRADEOFF_PLOT = "tradeoff_plot"
    STABILITY_PLOT = "stability_plot"
    IMBALANCE_IMPACT_PLOT = "imbalance_impact_plot"
    CS_VS_RESAMPLING_PLOT = "cs_vs_resampling_plot"
    METRICS_HEATMAP = "metrics_heatmap"


# Type alias for typer dataset argument
_DatasetArgument = Annotated[
    Optional[Dataset],
    typer.Argument(
        help=(
            "Identifier of the dataset to analyze. "
            "When omitted, all datasets are analyzed sequentially."
        ),
    ),
]

# Type alias for technique filter option
_TechniqueOption = Annotated[
    Optional[Technique],
    typer.Option(
        "--technique",
        "-t",
        help="Filter results by a specific technique (e.g., smote, rus).",
    ),
]

# Type alias for metric option
_MetricOption = Annotated[
    Metric,
    typer.Option(
        "--metric",
        "-m",
        help="Metric to visualize (e.g., accuracy_balanced, g_mean, f1_score).",
    ),
]

# Type alias for force overwrite option
_ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Force overwrite of existing artifacts.",
    ),
]

# Type alias for GPU option
_GpuOption = Annotated[
    bool,
    typer.Option(
        "--gpu",
        help="Enable GPU acceleration if available.",
    ),
]

# Type alias for locale option
_LocaleOption = Annotated[
    Locale,
    typer.Option(
        "--locale",
        "-l",
        help="Locale for generated artifacts (en_US or pt_BR).",
    ),
]

# Type alias for execution ID option
_ExecutionIdOption = Annotated[
    Optional[str],
    typer.Option(
        "--execution-id",
        "-e",
        help="Specific execution ID to analyze. If omitted, uses the latest execution.",
    ),
]


def _resolve_datasets(dataset: Dataset | None) -> list[Dataset]:
    """Resolve a single dataset to a list, defaulting to all datasets.

    Args:
        dataset: Optional single dataset.

    Returns:
        List of datasets to process.
    """
    if dataset is not None:
        return [dataset]
    return list(Dataset)


def _create_analysis_context(
    dataset: Dataset,
    analysis_name: str,
    force_overwrite: bool = False,
    use_gpu: bool = False,
    locale: Locale | None = None,
    execution_id: str | None = None,
) -> AnalysisPipelineContext:
    """Create an analysis pipeline context with injected dependencies.

    Args:
        dataset: The dataset to analyze.
        analysis_name: Name identifier for the analysis.
        force_overwrite: Whether to overwrite existing artifacts.
        use_gpu: Whether to use GPU acceleration.
        locale: Locale for internationalization. If None, uses default from settings.
        execution_id: Specific execution ID to analyze. If None, uses the latest.

    Returns:
        Configured AnalysisPipelineContext.
    """
    # Get storage and settings from container
    storage = container._storage()
    settings = container.settings()

    # Resolve locale: CLI option > settings default
    resolved_locale = locale if locale is not None else Locale(settings.locale)

    # Create translator
    translator = create_translator(resolved_locale)

    # Create repositories and evaluator
    predictions_repository = ModelPredictionsStorageRepository(storage=storage)
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)
    results_evaluator = ModelResultsEvaluatorImpl()

    return AnalysisPipelineContext(
        dataset=dataset,
        analysis_name=analysis_name,
        predictions_repository=predictions_repository,
        results_evaluator=results_evaluator,
        analysis_artifacts_repository=analysis_artifacts_repository,
        use_gpu=use_gpu,
        force_overwrite=force_overwrite,
        locale=resolved_locale,
        translator=translator,
        execution_id=execution_id,
    )


MODULE_NAME = "experiments.cli.analysis"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])


app = typer.Typer()


@app.command("summary")
def generate_summary_table(
    dataset: _DatasetArgument = None,
    technique: _TechniqueOption = None,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Generate a summary table of experiment results.

    Creates a LaTeX table with mean and standard deviation for each metric,
    sorted by balanced accuracy. Optionally filter by technique.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    # Get storage and settings from container
    storage = container._storage()

    # Create repositories
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)

    for ds in datasets:
        # Determine analysis name
        analysis_name = (
            f"summary_table_{technique.value}.tex" if technique else "summary_table.tex"
        )

        logger.info(f"Generating summary table for {ds.value}...")

        # Create factory with dependencies and configuration
        factory = SummaryTablePipelineFactory(
            analysis_artifacts_repository=analysis_artifacts_repository,
            technique_filter=technique,
        )

        # Build pipeline with artifact generator injected
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name({"dataset": ds.value}),
            artifact_generator=_generate_summary_table_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
            execution_id=execution_id,
        )

        # Execute pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            logger.success(f"Summary table saved for {ds.value}")
        else:
            logger.error(f"Failed to generate summary table for {ds.value}: {result.last_error()}")


@app.command("tradeoff")
def generate_tradeoff_plot(
    dataset: _DatasetArgument = None,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Generate a precision-sensitivity trade-off plot.

    Creates a scatter plot showing the trade-off between precision and sensitivity
    (recall) across different techniques and model types.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    # Get storage from container
    storage = container._storage()

    # Create repositories
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)

    for ds in datasets:
        analysis_name = "tradeoff_plot.png"

        logger.info(f"Generating trade-off plot for {ds.value}...")

        # Create factory with dependencies
        factory = TradeoffPlotPipelineFactory(
            analysis_artifacts_repository=analysis_artifacts_repository,
        )

        # Build pipeline with artifact generator injected
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name({"dataset": ds.value}),
            artifact_generator=_generate_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
            execution_id=execution_id,
        )

        # Execute pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            logger.success(f"Trade-off plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate trade-off plot for {ds.value}: {result.last_error()}"
            )


@app.command("stability")
def generate_stability_plot(
    dataset: _DatasetArgument = None,
    metric: _MetricOption = Metric.BALANCED_ACCURACY,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Generate a stability boxplot showing variance across seeds.

    Creates a boxplot visualization showing the distribution of a metric
    across different random seeds for each technique and model type.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    # Get storage from container
    storage = container._storage()

    # Create repositories
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)

    for ds in datasets:
        analysis_name = f"stability_plot_{metric.value}.png"

        logger.info(f"Generating stability plot for {ds.value} ({metric.value})...")

        # Create factory with dependencies and configuration
        factory = StabilityAnalysisPipelineFactory(
            analysis_artifacts_repository=analysis_artifacts_repository,
            metric=metric,
        )

        # Build pipeline with artifact generator injected
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name({"dataset": ds.value, "metric": metric.value}),
            artifact_generator=_generate_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
            execution_id=execution_id,
        )

        # Execute pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            logger.success(f"Stability plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate stability plot for {ds.value}: {result.last_error()}"
            )


@app.command("imbalance")
def generate_imbalance_impact_plot(
    dataset: _DatasetArgument = None,
    metric: _MetricOption = Metric.BALANCED_ACCURACY,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Generate an imbalance impact scatter plot.

    Creates a scatter plot showing how the imbalance ratio affects
    model performance for the specified metric.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    # Get storage from container
    storage = container._storage()

    # Create repositories
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)

    for ds in datasets:
        analysis_name = f"imbalance_impact_{metric.value}.png"

        logger.info(f"Generating imbalance impact plot for {ds.value}...")

        # Create factory with dependencies and configuration
        factory = ImbalanceImpactAnalysisPipelineFactory(
            analysis_artifacts_repository=analysis_artifacts_repository,
            metric=metric,
        )

        # Build pipeline with artifact generator injected
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name({"dataset": ds.value, "metric": metric.value}),
            artifact_generator=_generate_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
            execution_id=execution_id,
        )

        # Execute pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            logger.success(f"Imbalance impact plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate imbalance impact plot for {ds.value}: {result.last_error()}"
            )


@app.command("comparison")
def generate_cs_vs_resampling_plot(
    dataset: _DatasetArgument = None,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Generate a cost-sensitive vs resampling comparison plot.

    Creates a grouped bar chart comparing balanced accuracy for cost-sensitive
    methods (MetaCost, CS-SVM) vs resampling methods (SMOTE, RUS, SMOTE-Tomek).
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    # Get storage from container
    storage = container._storage()

    # Create repositories
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)

    for ds in datasets:
        analysis_name = "cs_vs_resampling_plot.png"

        logger.info(f"Generating CS vs resampling plot for {ds.value}...")

        # Create factory with dependencies
        factory = CostSensitiveVsResamplingComparisonPipelineFactory(
            analysis_artifacts_repository=analysis_artifacts_repository,
        )

        # Build pipeline with artifact generator injected
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name({"dataset": ds.value}),
            artifact_generator=_generate_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
            execution_id=execution_id,
        )

        # Execute pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            logger.success(f"CS vs resampling plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate CS vs resampling plot for {ds.value}: {result.last_error()}"
            )


@app.command("heatmap")
def generate_metrics_heatmap(
    dataset: _DatasetArgument = None,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Generate a metrics heatmap.

    Creates a heatmap visualization showing all metrics across techniques
    and model types, with alphabetically sorted rows and columns.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    # Get storage from container
    storage = container._storage()

    # Create repositories
    analysis_artifacts_repository = AnalysisArtifactsRepository(storage=storage)

    for ds in datasets:
        analysis_name = "metrics_heatmap.png"

        logger.info(f"Generating metrics heatmap for {ds.value}...")

        # Create factory with dependencies
        factory = MetricsHeatmapPipelineFactory(
            analysis_artifacts_repository=analysis_artifacts_repository,
        )

        # Build pipeline with artifact generator injected
        pipeline = factory.create_pipeline(
            name=factory.get_pipeline_name({"dataset": ds.value}),
            artifact_generator=_generate_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
            execution_id=execution_id,
        )

        # Execute pipeline
        result = executor.execute(
            pipeline=pipeline,
            initial_state={},  # type: ignore[arg-type]
            context=context,
        )

        if result.succeeded():
            logger.success(f"Metrics heatmap saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate metrics heatmap for {ds.value}: {result.last_error()}"
            )


@app.command("all")
def run_all_analyses(
    dataset: _DatasetArgument = None,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
    locale: _LocaleOption = Locale.PT_BR,
    execution_id: _ExecutionIdOption = None,
) -> None:
    """Run all analysis types sequentially.

    Generates all available analysis outputs (summary tables, plots, heatmaps)
    for the specified dataset(s).
    """
    generate_summary_table(
        dataset=dataset,
        technique=None,
        force=force,
        gpu=gpu,
        locale=locale,
        execution_id=execution_id,
    )
    generate_tradeoff_plot(
        dataset=dataset, force=force, gpu=gpu, locale=locale, execution_id=execution_id
    )
    generate_stability_plot(
        dataset=dataset,
        metric=Metric.BALANCED_ACCURACY,
        force=force,
        gpu=gpu,
        locale=locale,
        execution_id=execution_id,
    )
    generate_imbalance_impact_plot(
        dataset=dataset,
        metric=Metric.BALANCED_ACCURACY,
        force=force,
        gpu=gpu,
        locale=locale,
        execution_id=execution_id,
    )
    generate_cs_vs_resampling_plot(
        dataset=dataset, force=force, gpu=gpu, locale=locale, execution_id=execution_id
    )
    generate_metrics_heatmap(
        dataset=dataset, force=force, gpu=gpu, locale=locale, execution_id=execution_id
    )


if __name__ == "__main__":
    app()
