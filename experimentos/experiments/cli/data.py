"""CLI for data processing tasks."""

import sys

from joblib import Parallel, delayed
from loguru import logger
import typer
from typing_extensions import Annotated

from experiments.containers import container
from experiments.core.data import (
    DataProcessingPipelineFactory,
    Dataset,
)
from experiments.services.path_manager import PathManager
from experiments.utils.jobs import get_jobs_from_available_cpus
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.cli.data"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _process_single_dataset(
    path_manager: PathManager,
    use_gpu: bool,
    dataset: Dataset,
) -> tuple[Dataset, bool, str | None]:
    """Runs the preprocessing pipeline for a single dataset."""
    try:
        # Create pipeline factory with path manager as path provider
        factory = DataProcessingPipelineFactory(path_manager, use_gpu=use_gpu)

        # Create and run the pipeline for this dataset
        pipeline = factory.create(dataset)
        pipeline.run(dataset)

        return dataset, True, None

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to process dataset {dataset.display_name}: {exc}")
        return dataset, False, str(exc)


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
    path_manager = container.path_manager()
    resource_settings = container.settings().resources

    # Override use_gpu from settings if flag is set
    effective_use_gpu = use_gpu or resource_settings.use_gpu

    datasets_to_process = [dataset] if dataset is not None else list(Dataset)

    # Use path_manager for path checking in filter
    datasets_to_process = filter_items_for_processing(
        datasets_to_process,
        exists_fn=lambda ds: path_manager.get_interim_data_path(ds.id).exists(),
        prompt_fn=lambda ds: f"File '{path_manager.get_interim_data_path(ds.id)}' exists. Overwrite?",
        force=force,
        on_skip=lambda ds: logger.info(f"Skipping dataset {ds.display_name} per user choice."),
    )

    if not datasets_to_process:
        logger.info("No datasets selected for processing. Exiting.")
        return

    dataset_names = ", ".join(ds.display_name for ds in datasets_to_process)
    logger.info(f"Scheduling preprocessing for: {dataset_names}")

    n_jobs = get_jobs_from_available_cpus(jobs)

    # Pass path_manager to workers
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(path_manager, effective_use_gpu, ds)
        for ds in datasets_to_process
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.display_name for ds in failed)
        logger.error(f"Processing failed for: {failed_names}")
        raise typer.Exit(code=1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    # Ensure pickle compatibility
    for _func in [_process_single_dataset, main]:
        _func.__module__ = MODULE_NAME
    app()
