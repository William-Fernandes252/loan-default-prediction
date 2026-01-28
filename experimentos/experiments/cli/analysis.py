"""CLI for analyzing experimental results.

This module provides commands for various analysis types on the resultant models
from the experiments. Each command generates visualizations and reports to help
understand model performance, stability, and the effects of different techniques.

The CLI follows the "Append Strategy" pattern:
- Pipeline factories build the core business logic (data -> metrics -> visualization)
- The CLI appends persistence tasks at runtime based on the analysis type
"""

import enum
import sys
from typing import Optional

from loguru import logger
import typer
from typing_extensions import Annotated

from experiments.containers import container
from experiments.core.analysis import Locale
from experiments.core.analysis.metrics import Metric
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines.execution import PipelineExecutor
from experiments.pipelines.analysis.factory import (
    build_cs_vs_resampling_pipeline,
    build_imbalance_impact_pipeline,
    build_metrics_heatmap_pipeline,
    build_stability_pipeline,
    build_summary_table_pipeline,
    build_tradeoff_plot_pipeline,
)
from experiments.pipelines.analysis.pipeline import AnalysisPipelineContext
from experiments.pipelines.analysis.tasks.persistence import (
    save_figure_artifact,
    save_table_artifact,
)
from experiments.services.analysis_artifacts_repository import AnalysisArtifactsRepository
from experiments.services.model_predictions_repository import ModelPredictionsStorageRepository
from experiments.services.model_results_evaluator import ModelResultsEvaluatorImpl
from experiments.services.translator import create_translator


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
) -> AnalysisPipelineContext:
    """Create an analysis pipeline context with injected dependencies.

    Args:
        dataset: The dataset to analyze.
        analysis_name: Name identifier for the analysis.
        force_overwrite: Whether to overwrite existing artifacts.
        use_gpu: Whether to use GPU acceleration.
        locale: Locale for internationalization. If None, uses default from settings.

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
        translator=translator,
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
) -> None:
    """Generate a summary table of experiment results.

    Creates a LaTeX table with mean and standard deviation for each metric,
    sorted by balanced accuracy. Optionally filter by technique.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    for ds in datasets:
        # Determine analysis name
        analysis_name = f"summary_table_{technique.value}" if technique else "summary_table"

        logger.info(f"Generating summary table for {ds.value}...")

        # Build pipeline from factory
        pipeline = build_summary_table_pipeline(technique=technique)

        # Append persistence step (CLI responsibility)
        pipeline.add_step(
            name="SaveTableArtifact",
            task=save_table_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
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
) -> None:
    """Generate a precision-sensitivity trade-off plot.

    Creates a scatter plot showing the trade-off between precision and sensitivity
    (recall) across different techniques and model types.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    for ds in datasets:
        analysis_name = "tradeoff_plot"

        logger.info(f"Generating trade-off plot for {ds.value}...")

        # Build pipeline from factory
        pipeline = build_tradeoff_plot_pipeline()

        # Append persistence step (CLI responsibility)
        pipeline.add_step(
            name="SaveFigureArtifact",
            task=save_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
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
) -> None:
    """Generate a stability boxplot showing variance across seeds.

    Creates a boxplot visualization showing the distribution of a metric
    across different random seeds for each technique and model type.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    for ds in datasets:
        analysis_name = f"stability_plot_{metric.value}"

        logger.info(f"Generating stability plot for {ds.value} ({metric.value})...")

        # Build pipeline from factory
        pipeline = build_stability_pipeline(metric=metric)

        # Append persistence step (CLI responsibility)
        pipeline.add_step(
            name="SaveFigureArtifact",
            task=save_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
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
) -> None:
    """Generate an imbalance impact scatter plot.

    Creates a scatter plot showing how the imbalance ratio affects
    model performance for the specified metric.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    for ds in datasets:
        analysis_name = f"imbalance_impact_{metric.value}"

        logger.info(f"Generating imbalance impact plot for {ds.value}...")

        # Build pipeline from factory
        pipeline = build_imbalance_impact_pipeline(metric=metric)

        # Append persistence step (CLI responsibility)
        pipeline.add_step(
            name="SaveFigureArtifact",
            task=save_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
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
) -> None:
    """Generate a cost-sensitive vs resampling comparison plot.

    Creates a grouped bar chart comparing balanced accuracy for cost-sensitive
    methods (MetaCost, CS-SVM) vs resampling methods (SMOTE, RUS, SMOTE-Tomek).
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    for ds in datasets:
        analysis_name = "cs_vs_resampling_plot"

        logger.info(f"Generating CS vs resampling plot for {ds.value}...")

        # Build pipeline from factory
        pipeline = build_cs_vs_resampling_pipeline()

        # Append persistence step (CLI responsibility)
        pipeline.add_step(
            name="SaveFigureArtifact",
            task=save_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
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
) -> None:
    """Generate a metrics heatmap.

    Creates a heatmap visualization showing all metrics across techniques
    and model types, with alphabetically sorted rows and columns.
    """
    datasets = _resolve_datasets(dataset)
    executor = PipelineExecutor(max_workers=1)

    for ds in datasets:
        analysis_name = "metrics_heatmap"

        logger.info(f"Generating metrics heatmap for {ds.value}...")

        # Build pipeline from factory
        pipeline = build_metrics_heatmap_pipeline()

        # Append persistence step (CLI responsibility)
        pipeline.add_step(
            name="SaveFigureArtifact",
            task=save_figure_artifact,  # type: ignore[arg-type]
        )

        # Create context with dependencies
        context = _create_analysis_context(
            dataset=ds,
            analysis_name=analysis_name,
            force_overwrite=force,
            use_gpu=gpu,
            locale=locale,
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
) -> None:
    """Run all analysis types sequentially.

    Generates all available analysis outputs (summary tables, plots, heatmaps)
    for the specified dataset(s).
    """
    generate_summary_table(dataset=dataset, technique=None, force=force, gpu=gpu, locale=locale)
    generate_tradeoff_plot(dataset=dataset, force=force, gpu=gpu, locale=locale)
    generate_stability_plot(
        dataset=dataset, metric=Metric.BALANCED_ACCURACY, force=force, gpu=gpu, locale=locale
    )
    generate_imbalance_impact_plot(
        dataset=dataset, metric=Metric.BALANCED_ACCURACY, force=force, gpu=gpu, locale=locale
    )
    generate_cs_vs_resampling_plot(dataset=dataset, force=force, gpu=gpu, locale=locale)
    generate_metrics_heatmap(dataset=dataset, force=force, gpu=gpu, locale=locale)


if __name__ == "__main__":
    app()
