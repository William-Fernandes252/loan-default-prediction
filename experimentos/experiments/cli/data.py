"""CLI for data processing tasks."""

import sys
from typing import Annotated, cast

from joblib import Parallel, delayed
from loguru import logger
import typer

from experiments.containers import NewContainer
from experiments.core.data_new import (
    Dataset,
)
from experiments.lib.pipelines import PipelineException, PipelineExecutionResult
from experiments.lib.pipelines.errors import PipelineInterruption
from experiments.lib.pipelines.executor import ErrorAction
from experiments.pipelines.data.context import DataPipelineContext
from experiments.pipelines.data.factory import DataProcessingPipeline, DataProcessingPipelineSteps

MODULE_NAME = "experiments.cli.data"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _prompt_user_for_confirmation(
    step_name: str, exception: PipelineException | PipelineInterruption
) -> ErrorAction:
    """Prompt the user for confirmation on pipeline interruption."""

    if step_name == DataProcessingPipelineSteps.CHECK_ALREADY_PROCESSED.value and isinstance(
        exception, PipelineInterruption
    ):
        return (
            ErrorAction.IGNORE
            if typer.confirm(str(exception), default=False)
            else ErrorAction.ABORT
        )

    return ErrorAction.ABORT


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

    dataset_names = ", ".join(ds.display_name for ds in datasets_to_process)
    logger.info(f"Scheduling preprocessing for: {dataset_names}")

    executor = container.executor()
    pipelines = [
        container.data_processing_pipeline_factory().create(ds, effective_use_gpu, force)
        for ds in datasets_to_process
    ]
    n_jobs = jobs or container.resource_calculator().compute_safe_jobs(
        dataset_size_gb=sum(
            data_repository.get_size_in_bytes(ds) / 1e9 for ds in datasets_to_process
        ),
    )

    def run_data_pipeline(pipeline: DataProcessingPipeline) -> PipelineExecutionResult:
        with logger.contextualize(dataset=pipeline.context.dataset.display_name):
            return executor.execute(
                pipeline,
                {},
                error_handlers={
                    DataProcessingPipelineSteps.CHECK_ALREADY_PROCESSED.value: _prompt_user_for_confirmation
                },
            )

    results = cast(
        list[PipelineExecutionResult[object, DataPipelineContext]],
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_data_pipeline)(pipeline) for pipeline in pipelines
        ),
    )

    failed = [result for result in results if not result.succeeded()]
    if failed:
        for result in failed:
            error_details = result.last_error()
            if error_details:
                logger.error(
                    f"Dataset {result.context.dataset.display_name} failed: {error_details}"
                )
        raise typer.Exit(code=1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    app()
