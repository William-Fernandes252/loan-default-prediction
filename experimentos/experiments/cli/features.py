"""CLI for features and data split tasks."""

import sys

from joblib import Parallel, delayed
from loguru import logger
import typer
from typing_extensions import Annotated

from experiments.containers import container
from experiments.core.data import Dataset
from experiments.core.modeling.features import extract_features_and_target
from experiments.services.storage_manager import StorageManager
from experiments.utils.jobs import get_jobs_from_available_cpus
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.cli.features"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _artifacts_exist(storage_manager: StorageManager, dataset: Dataset) -> bool:
    uris = storage_manager.get_feature_uris(dataset.id)
    storage = storage_manager.storage
    return any(storage.exists(uri) for uri in uris.values())


def _process_single_dataset(
    dataset: Dataset,
) -> tuple[Dataset, bool, str | None]:
    """Orchestrates the feature extraction for a single dataset."""
    try:
        logger.info(f"Preparing features (X/y) for: {dataset.display_name}")

        storage_manager = container.storage_manager()
        storage = storage_manager.storage
        input_uri = storage_manager.get_interim_data_uri(dataset.id)
        if not storage.exists(input_uri):
            msg = f"File not found: {input_uri}. Run 'experiments.cli.data' first."
            raise FileNotFoundError(msg)

        # 1. Load Intermediate Data
        logger.info(f"Loading data from {input_uri}")
        df = storage.read_parquet(input_uri)

        # 2. Core Logic
        X_final, y_final = extract_features_and_target(df)

        # 3. Save Artifacts
        artifacts = storage_manager.get_feature_uris(dataset.id)

        logger.info(f"Saving X (shape={X_final.shape}) and y (shape={y_final.shape})...")
        storage.write_parquet(X_final, artifacts["X"])
        storage.write_parquet(y_final, artifacts["y"])

        logger.success(f"Processed data saved for {dataset.display_name}")
        return dataset, True, None

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to process features for {dataset.display_name}: {exc}")
        return dataset, False, str(exc)


@app.command(name="prepare")
def main(
    dataset: Annotated[
        Dataset | None,
        typer.Argument(help="Dataset to be processed."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing files."),
    ] = False,
    jobs: Annotated[
        int | None,
        typer.Option(
            "--jobs",
            "-j",
            min=1,
            help=(
                "Number of parallel workers. Defaults to detected CPUs. "
                "Values above the dataset count are clamped."
            ),
        ),
    ] = None,
):
    """Prepares full X matrices and y vectors for training."""
    # Resolve dependencies from container
    storage_manager = container.storage_manager()

    datasets = [dataset] if dataset is not None else list(Dataset)

    # Filter using storage_manager
    datasets = filter_items_for_processing(
        datasets,
        exists_fn=lambda ds: _artifacts_exist(storage_manager, ds),
        prompt_fn=lambda ds: f"Features for '{ds.display_name}' exist. Overwrite?",
        force=force,
        on_skip=lambda ds: logger.info(f"Skipping dataset {ds.display_name} per user choice."),
    )

    if not datasets:
        logger.info("No dataset selected.")
        return

    dataset_names = ", ".join(ds.display_name for ds in datasets)
    logger.info(f"Scheduling feature preparation for: {dataset_names}")

    n_jobs = get_jobs_from_available_cpus(jobs)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ds) for ds in datasets
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.display_name for ds in failed)
        logger.error(f"Feature preparation failed for: {failed_names}")
        raise typer.Exit(code=1)

    logger.success("All requested feature artifacts generated successfully.")


if __name__ == "__main__":
    for _func in [_process_single_dataset, main]:
        _func.__module__ = MODULE_NAME
    app()
