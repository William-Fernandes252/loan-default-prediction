"""CLI for data processing tasks."""

import sys
from typing import Annotated

from loguru import logger
import typer

from experiments.containers import container
from experiments.core.data import (
    Dataset,
)

MODULE_NAME = "experiments.cli.data"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


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
    datasets_to_process = [dataset] if dataset is not None else list(Dataset)
    if not datasets_to_process:
        logger.info("No datasets selected for processing. Exiting.")
        return

    dataset_names = ", ".join(ds.value for ds in datasets_to_process)
    logger.info(f"Scheduling preprocessing for: {dataset_names}")

    data_manager = container.data_manager()
    errors = data_manager.process_datasets(
        datasets=datasets_to_process,
        force_overwrite=force,
        use_gpu=use_gpu,
        workers=jobs,
    )
    if errors:
        for dataset, error in errors:
            logger.error(f"Error processing dataset {dataset}: {error}")
        raise typer.Exit(code=1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    app()
