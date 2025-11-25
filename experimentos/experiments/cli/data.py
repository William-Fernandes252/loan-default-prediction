"""CLI for data processing tasks."""

import sys
from typing import Any

from joblib import Parallel, delayed
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.context import AppConfig, Context
from experiments.core.data import (
    CorporateCreditProcessor,
    DataProcessor,
    Dataset,
    LendingClubProcessor,
    TaiwanCreditProcessor,
)
from experiments.utils.jobs import get_jobs_from_available_cpus
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.cli.data"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


# Factory pattern to get the right processor
def get_processor(ctx: Context, dataset: Dataset) -> DataProcessor:
    PROCESSORS = {
        Dataset.LENDING_CLUB: LendingClubProcessor(use_gpu=ctx.cfg.use_gpu),
        Dataset.CORPORATE_CREDIT_RATING: CorporateCreditProcessor(use_gpu=ctx.cfg.use_gpu),
        Dataset.TAIWAN_CREDIT: TaiwanCreditProcessor(use_gpu=ctx.cfg.use_gpu),
    }
    return PROCESSORS[dataset]


def _process_single_dataset(ctx: Context, dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    """Runs the preprocessing pipeline for a single dataset."""
    try:
        ctx.logger.info(f"Processing dataset {dataset.display_name}...")

        raw_data_path = ctx.get_raw_data_path(dataset.id)
        output_path = ctx.get_interim_data_path(dataset.id)

        ctx.logger.info(f"Loading raw data from {raw_data_path}...")
        read_csv_kwargs: dict[str, Any] = {"low_memory": False, "use_pyarrow": True}
        read_csv_kwargs.update(dataset.get_extra_params())

        raw_data = pl.read_csv(raw_data_path, **read_csv_kwargs)
        ctx.logger.info("Raw data loaded.")

        # Get Strategy via Factory
        processor = get_processor(ctx, dataset)

        # Run transformations
        processed_data = processor.process(raw_data)
        processed_data = processor.sanitize(processed_data)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.logger.info(f"Saving processed data to {output_path}...")
        processed_data.write_parquet(output_path)

        ctx.logger.success(f"Processing dataset {dataset.display_name} complete.")
        return dataset, True, None

    except Exception as exc:  # noqa: BLE001
        ctx.logger.exception(f"Failed to process dataset {dataset.display_name}: {exc}")
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
    ctx = Context(AppConfig(use_gpu=use_gpu))

    datasets_to_process = [dataset] if dataset is not None else list(Dataset)

    # Use context for path checking in filter
    datasets_to_process = filter_items_for_processing(
        datasets_to_process,
        exists_fn=lambda ds: ctx.get_interim_data_path(ds.id).exists(),
        prompt_fn=lambda ds: f"File '{ctx.get_interim_data_path(ds.id)}' exists. Overwrite?",
        force=force,
        on_skip=lambda ds: ctx.logger.info(f"Skipping dataset {ds.display_name} per user choice."),
    )

    if not datasets_to_process:
        ctx.logger.info("No datasets selected for processing. Exiting.")
        return

    dataset_names = ", ".join(ds.display_name for ds in datasets_to_process)
    ctx.logger.info(f"Scheduling preprocessing for: {dataset_names}")

    n_jobs = get_jobs_from_available_cpus(jobs)

    # Pass ctx to workers
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ctx, ds) for ds in datasets_to_process
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.display_name for ds in failed)
        ctx.logger.error(f"Processing failed for: {failed_names}")
        raise typer.Exit(code=1)

    ctx.logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    # Ensure pickle compatibility
    for _func in [_process_single_dataset, main]:
        _func.__module__ = MODULE_NAME
    app()
