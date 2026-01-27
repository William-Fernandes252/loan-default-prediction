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
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import Technique
from experiments.lib.pipelines.execution import PipelineExecutor
from experiments.pipelines.analysis.factory import (
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


class AnalysisType(enum.StrEnum):
    """Available analysis types."""

    SUMMARY_TABLE = "summary_table"
    TRADEOFF_PLOT = "tradeoff_plot"


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
) -> AnalysisPipelineContext:
    """Create an analysis pipeline context with injected dependencies.

    Args:
        dataset: The dataset to analyze.
        analysis_name: Name identifier for the analysis.
        force_overwrite: Whether to overwrite existing artifacts.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        Configured AnalysisPipelineContext.
    """
    # Get storage from container
    storage = container._storage()

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


@app.command("all")
def run_all_analyses(
    dataset: _DatasetArgument = None,
    force: _ForceOption = False,
    gpu: _GpuOption = False,
) -> None:
    """Run all analysis types sequentially.

    Generates both summary tables and trade-off plots for the specified
    dataset(s).
    """
    generate_summary_table(dataset=dataset, technique=None, force=force, gpu=gpu)
    generate_tradeoff_plot(dataset=dataset, force=force, gpu=gpu)


if __name__ == "__main__":
    app()
