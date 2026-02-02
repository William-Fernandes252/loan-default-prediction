"""CLI for analyzing experimental results.

This module provides commands for various analysis types on the resultant models
from the experiments. Each command generates visualizations and reports to help
understand model performance, stability, and the effects of different techniques.

The CLI delegates all analysis work to the PredictionsAnalyzer service,
keeping commands simple and focused on argument parsing and result logging.
"""

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
from experiments.services.predictions_analyzer import AnalysisParams, AnalysisType

_DatasetArgument = Annotated[
    Optional[Dataset],
    typer.Argument(
        help=(
            "Identifier of the dataset to analyze. "
            "When omitted, all datasets are analyzed sequentially."
        ),
    ),
]
"""Type alias for dataset argument."""

_TechniqueOption = Annotated[
    Optional[Technique],
    typer.Option(
        "--technique",
        "-t",
        help="Filter results by a specific technique (e.g., smote, rus).",
    ),
]
"""Type alias for technique option."""

_MetricOption = Annotated[
    Metric,
    typer.Option(
        "--metric",
        "-m",
        help="Metric to visualize (e.g., accuracy_balanced, g_mean, f1_score).",
    ),
]
"""Type alias for metric option."""

_ForceOption = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Force overwrite of existing artifacts.",
    ),
]
"""Type alias for force overwrite option."""

_GpuOption = Annotated[
    bool,
    typer.Option(
        "--gpu",
        help="Enable GPU acceleration if available.",
    ),
]
"""Type alias for GPU option."""

_LocaleOption = Annotated[
    Locale,
    typer.Option(
        "--locale",
        "-l",
        help="Locale for generated artifacts (en_US or pt_BR).",
    ),
]
"""Type alias for locale option."""

_ExecutionIdOption = Annotated[
    Optional[str],
    typer.Option(
        "--execution-id",
        "-e",
        help="Specific execution ID to analyze. If omitted, uses the latest execution.",
    ),
]
"""Type alias for execution ID option."""


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
    analyzer = container.predictions_analyzer()

    for ds in _resolve_datasets(dataset):
        logger.info(f"Generating summary table for {ds.value}...")

        result = analyzer.run_analysis(
            analysis_type=AnalysisType.SUMMARY_TABLE,
            params=AnalysisParams(
                dataset=ds,
                locale=locale,
                force_overwrite=force,
                use_gpu=gpu,
                execution_id=execution_id,
                technique_filter=technique,
            ),
        )

        if result.succeeded():
            logger.success(f"Summary table saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate summary table for {ds.value}: {result.error_message}"
            )


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
    analyzer = container.predictions_analyzer()

    for ds in _resolve_datasets(dataset):
        logger.info(f"Generating trade-off plot for {ds.value}...")

        result = analyzer.run_analysis(
            analysis_type=AnalysisType.TRADEOFF_PLOT,
            params=AnalysisParams(
                dataset=ds,
                locale=locale,
                force_overwrite=force,
                use_gpu=gpu,
                execution_id=execution_id,
            ),
        )

        if result.succeeded():
            logger.success(f"Trade-off plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate trade-off plot for {ds.value}: {result.error_message}"
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
    analyzer = container.predictions_analyzer()

    for ds in _resolve_datasets(dataset):
        logger.info(f"Generating stability plot for {ds.value} ({metric.value})...")

        result = analyzer.run_analysis(
            analysis_type=AnalysisType.STABILITY_PLOT,
            params=AnalysisParams(
                dataset=ds,
                locale=locale,
                force_overwrite=force,
                use_gpu=gpu,
                execution_id=execution_id,
                metric=metric,
            ),
        )

        if result.succeeded():
            logger.success(f"Stability plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate stability plot for {ds.value}: {result.error_message}"
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
    analyzer = container.predictions_analyzer()

    for ds in _resolve_datasets(dataset):
        logger.info(f"Generating imbalance impact plot for {ds.value}...")

        result = analyzer.run_analysis(
            analysis_type=AnalysisType.IMBALANCE_IMPACT_PLOT,
            params=AnalysisParams(
                dataset=ds,
                locale=locale,
                force_overwrite=force,
                use_gpu=gpu,
                execution_id=execution_id,
                metric=metric,
            ),
        )

        if result.succeeded():
            logger.success(f"Imbalance impact plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate imbalance impact plot for {ds.value}: {result.error_message}"
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
    analyzer = container.predictions_analyzer()

    for ds in _resolve_datasets(dataset):
        logger.info(f"Generating CS vs resampling plot for {ds.value}...")

        result = analyzer.run_analysis(
            analysis_type=AnalysisType.CS_VS_RESAMPLING_PLOT,
            params=AnalysisParams(
                dataset=ds,
                locale=locale,
                force_overwrite=force,
                use_gpu=gpu,
                execution_id=execution_id,
            ),
        )

        if result.succeeded():
            logger.success(f"CS vs resampling plot saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate CS vs resampling plot for {ds.value}: {result.error_message}"
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
    analyzer = container.predictions_analyzer()

    for ds in _resolve_datasets(dataset):
        logger.info(f"Generating metrics heatmap for {ds.value}...")

        result = analyzer.run_analysis(
            analysis_type=AnalysisType.METRICS_HEATMAP,
            params=AnalysisParams(
                dataset=ds,
                locale=locale,
                force_overwrite=force,
                use_gpu=gpu,
                execution_id=execution_id,
            ),
        )

        if result.succeeded():
            logger.success(f"Metrics heatmap saved for {ds.value}")
        else:
            logger.error(
                f"Failed to generate metrics heatmap for {ds.value}: {result.error_message}"
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
