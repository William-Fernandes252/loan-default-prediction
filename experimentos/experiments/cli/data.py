"""CLI for data processing tasks."""

import sys
from typing import Annotated

from loguru import logger
import typer

from experiments.config.logging import LoggingObserver
from experiments.containers import NewContainer
from experiments.core.data import (
    Dataset,
)
from experiments.lib.pipelines import (
    Action,
    PipelineExecutor,
    PipelineStatus,
    TaskResult,
    TaskStatus,
)
from experiments.lib.pipelines.lifecycle import IgnoreAllObserver
from experiments.pipelines.data import (
    DataPipelineContext,
    DataPipelineState,
    DataProcessingPipeline,
    DataProcessingPipelineSteps,
)

MODULE_NAME = "experiments.cli.data"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


class AbortIfAlreadyProcessedObserver(IgnoreAllObserver[DataPipelineState, DataPipelineContext]):
    """Observer that aborts the pipeline if processed data already exists."""

    def __init__(self, force: bool = False) -> None:
        """Initialize the observer.

        Args:
            force: If True, overwrite existing processed data.
        """
        self._force = force

    def on_step_finish(
        self,
        pipeline: DataProcessingPipeline,
        step_name: str,
        result: TaskResult[DataPipelineState],
    ) -> Action:
        """Handle step completion and control flow based on results.

        For the CHECK_ALREADY_PROCESSED step, decides whether to
        continue (overwrite) or abort based on the --force flag.
        """
        # Handle the check for existing processed data
        if step_name == DataProcessingPipelineSteps.CHECK_ALREADY_PROCESSED.value:
            if result.status == TaskStatus.SKIPPED:
                if self._force:
                    logger.info(f"[{pipeline.name}] {result.message} Overwriting (--force)")
                    return Action.PROCEED
                else:
                    logger.warning(
                        f"[{pipeline.name}] {result.message} Aborting (use --force to overwrite)"
                    )
                    return Action.ABORT

        return Action.PROCEED


@app.command(name="process")
def main(
    dataset: Annotated[
        Dataset | None,
        typer.Argument(
            help=(
                "Identifier of the dataset to process. "
                "When omitted, all datasets are processed in parallel."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing processed files without prompting.",
        ),
    ] = False,
    jobs: Annotated[
        int | None,
        typer.Option(
            "--jobs",
            "-j",
            min=1,
            help=(
                "Number of parallel workers. Defaults to the available CPU count. "
                "Values greater than the number of datasets are clamped."
            ),
        ),
    ] = None,
    use_gpu: Annotated[
        bool,
        typer.Option(
            "--use-gpu",
            "-g",
            help="Utilize GPU acceleration if available during processing.",
        ),
    ] = False,
):
    """Processes one or all datasets."""
    # Resolve dependencies from container
    container = NewContainer()

    data_repository = container.data_repository()
    resource_settings = container.config.resources

    # Override use_gpu from settings if flag is set
    effective_use_gpu = use_gpu or resource_settings.use_gpu()

    datasets_to_process = [dataset] if dataset is not None else list(Dataset)

    if not datasets_to_process:
        logger.info("No datasets selected for processing. Exiting.")
        return

    dataset_names = ", ".join(ds.value for ds in datasets_to_process)
    logger.info(f"Scheduling preprocessing for: {dataset_names}")

    # Calculate number of workers
    n_jobs = jobs or container.resource_calculator().compute_safe_jobs(
        dataset_size_gb=sum(
            data_repository.get_size_in_bytes(ds) / 1e9 for ds in datasets_to_process
        ),
    )

    # Create pipelines
    pipelines = [
        container.data_processing_pipeline_factory().create(ds, effective_use_gpu, force)
        for ds in datasets_to_process
    ]

    # Create observers
    observers = {AbortIfAlreadyProcessedObserver(force=force), LoggingObserver()}

    # Create executor with parallel workers and observer
    executor = PipelineExecutor(
        max_workers=n_jobs,
        observers=observers,
        default_action=Action.ABORT,
    )

    # Schedule all pipelines with empty initial state
    for pipeline in pipelines:
        initial_state: DataPipelineState = {
            "raw_data": None,
            "interim_data": None,
            "X_final": None,
            "y_final": None,
        }
        executor.schedule(pipeline, initial_state)

    executor.start()

    results = executor.wait()
    failed = [result for result in results if not result.succeeded()]
    if failed:
        for result in failed:
            if result.status == PipelineStatus.ABORTED:
                logger.warning(f"Pipeline {result.pipeline_name} was aborted")
            elif result.status == PipelineStatus.PANICKED:
                logger.error(f"Pipeline {result.pipeline_name} panicked")
            else:
                error_details = result.last_error()
                if error_details:
                    logger.error(f"Pipeline {result.pipeline_name} failed: {error_details}")
        raise typer.Exit(code=1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    app()
