"""CLI for data processing tasks."""

from pathlib import Path
import sys
from typing import Any

from joblib import Parallel, cpu_count, delayed
from loguru import logger
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from experiments.core.data import Dataset, get_processor
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.cli.data"

if __name__ == "__main__":
    # Fix for joblib pickling when running via `python -m experiments.cli.data`
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def get_processed_path(dataset: Dataset) -> Path:
    """Returns the processed data file path for the given dataset."""
    return INTERIM_DATA_DIR / f"{dataset.value}.parquet"


def _filter_datasets_for_processing(
    datasets: list[Dataset],
    *,
    force: bool,
) -> list[Dataset]:
    """Return the subset of datasets approved for processing."""
    return filter_items_for_processing(
        datasets,
        exists_fn=lambda ds: get_processed_path(ds).exists(),
        prompt_fn=lambda ds: f"Processed file '{get_processed_path(ds)}' already exists. Overwrite it?",
        force=force,
        on_skip=lambda ds: logger.info(f"Skipping dataset {ds} per user choice."),
    )


def _process_single_dataset(dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    """Runs the preprocessing pipeline for a single dataset."""
    try:
        logger.info(f"Processing dataset {dataset}...")

        raw_data_path = dataset.get_path(RAW_DATA_DIR)
        output_path = get_processed_path(dataset)

        # 1. Load Data
        # We keep the file reading here to maintain control over I/O options
        # (like low_memory) before passing the DataFrame to the logic core.
        logger.info(f"Loading raw data from {raw_data_path}...")
        read_csv_kwargs: dict[str, Any] = {"low_memory": False, "use_pyarrow": True}
        read_csv_kwargs.update(dataset.get_extra_params())

        raw_data = pl.read_csv(raw_data_path, **read_csv_kwargs)
        logger.info("Raw data loaded.")

        # 2. Get Strategy and Process
        # This is the key refactor: we use the factory to get the strategy
        processor = get_processor(dataset)

        # Run the specific transformation logic (Lazy execution handled internally by processor)
        processed_data = processor.process(raw_data)

        # Run the shared sanitization logic (Inf/NaN handling) defined in the Base Class
        processed_data = processor.sanitize(processed_data)

        # 3. Save Data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed data to {output_path}...")
        processed_data.write_parquet(output_path)

        logger.success(f"Processing dataset {dataset} complete.")
        return dataset, True, None

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to process dataset {dataset}: {exc}")
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
):
    """Processes one or all datasets."""

    datasets_to_process = [dataset] if dataset is not None else list(Dataset)
    datasets_to_process = _filter_datasets_for_processing(
        datasets_to_process,
        force=force,
    )

    if not datasets_to_process:
        logger.info("No datasets selected for processing. Exiting.")
        return

    dataset_names = ", ".join(ds.value for ds in datasets_to_process)
    logger.info(
        f"Scheduling preprocessing for {len(datasets_to_process)} dataset(s): {dataset_names}"
    )

    available_cpus = cpu_count() or 1
    requested_jobs = jobs if jobs is not None else available_cpus
    n_jobs = min(len(datasets_to_process), max(1, requested_jobs))

    if jobs is not None:
        logger.info(f"Using {n_jobs} parallel job(s) managed by joblib (user requested {jobs}).")
    else:
        logger.info(
            f"Using {n_jobs} parallel job(s) managed by joblib (detected {available_cpus} CPU(s))."
        )

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ds) for ds in datasets_to_process
    )

    failed: list[Dataset] = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.value for ds in failed)
        logger.error(f"Processing failed for the following dataset(s): {failed_names}")
        raise typer.Exit(code=1)

    logger.success("All requested datasets processed successfully.")


if __name__ == "__main__":
    # Ensure the functions we are pickling are associated with the correct module name
    for _func in [
        _process_single_dataset,
        main,
    ]:
        _func.__module__ = MODULE_NAME

    app()
