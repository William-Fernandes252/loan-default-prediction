from pathlib import Path
import sys

from joblib import Parallel, cpu_count, delayed
from loguru import logger
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, Dataset
from experiments.core.modeling.features import extract_features_and_target
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.cli.features"

if __name__ == "__main__":
    # Fix for joblib pickling when running via `python -m experiments.cli.features`
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()

_BASE_PATH = PROCESSED_DATA_DIR


def get_processed_input_path(dataset: Dataset) -> Path:
    """
    Reconstructs the path to the processed data (output of the data command).
    We define it here or import from config to avoid dependencies on other CLI scripts.
    """
    return INTERIM_DATA_DIR / f"{dataset.value}.parquet"


def _get_artifact_paths(dataset: Dataset) -> dict[str, Path]:
    """
    Returns the output paths for the complete data (without split).
    Splitting will be done dynamically during training to support the 30 runs.
    """
    return {
        "X": _BASE_PATH / f"{dataset.value}_X.parquet",
        "y": _BASE_PATH / f"{dataset.value}_y.parquet",
    }


def _artifacts_exist(dataset: Dataset) -> bool:
    return any(path.exists() for path in _get_artifact_paths(dataset).values())


def _filter_datasets_for_processing(
    datasets: list[Dataset],
    *,
    force: bool,
) -> list[Dataset]:
    return filter_items_for_processing(
        datasets,
        exists_fn=_artifacts_exist,
        prompt_fn=lambda ds: (
            "Processed features for '{name}' already exist ({paths}). Overwrite?".format(
                name=ds.value,
                paths=", ".join(str(path) for path in _get_artifact_paths(ds).values()),
            )
        ),
        force=force,
        on_skip=lambda ds: logger.info(f"Skipping dataset {ds} per user choice."),
    )


def _ensure_artifact_directories(artifacts: dict[str, Path]) -> None:
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _process_single_dataset(dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    """
    Orchestrates the feature extraction for a single dataset.
    Loads data -> Calls Core Logic -> Saves data.
    """
    try:
        logger.info(f"Preparing features (X/y) for: {dataset.value}")

        input_path = get_processed_input_path(dataset)
        if not input_path.exists():
            msg = f"File not found: {input_path}. Run 'experiments.cli.data' first."
            raise FileNotFoundError(msg)

        # 1. Load Intermediate Data
        logger.info(f"Loading data from {input_path}")
        df = pl.read_parquet(input_path, use_pyarrow=True)

        # 2. Core Logic: Split X and y and Sanitize
        X_final, y_final = extract_features_and_target(df)

        # 3. Save Artifacts
        artifacts = _get_artifact_paths(dataset)
        _ensure_artifact_directories(artifacts)

        logger.info(f"Saving X (shape={X_final.shape}) and y (shape={y_final.shape})...")
        X_final.write_parquet(artifacts["X"])
        y_final.write_parquet(artifacts["y"])

        logger.success(f"Processed data saved in: {artifacts['X'].parent}")
        return dataset, True, None

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to process features for {dataset}: {exc}")
        return dataset, False, str(exc)


@app.command()
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
    """
    Prepares full X matrices and y vectors for training.
    Does not perform splitting or scaling (this happens in the experimental loop).
    """
    datasets = [dataset] if dataset is not None else list(Dataset)
    datasets = _filter_datasets_for_processing(datasets, force=force)

    if not datasets:
        logger.info("No dataset selected.")
        return

    dataset_names = ", ".join(ds.value for ds in datasets)
    logger.info(f"Scheduling feature preparation for {len(datasets)} dataset(s): {dataset_names}")

    available_cpus = cpu_count() or 1
    requested_jobs = jobs if jobs is not None else available_cpus
    n_jobs = min(len(datasets), max(1, requested_jobs))

    if jobs is not None:
        logger.info(f"Using {n_jobs} parallel job(s) (user requested {jobs}).")
    else:
        logger.info(f"Using {n_jobs} parallel job(s) (detected {available_cpus} CPU(s)).")

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ds) for ds in datasets
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.value for ds in failed)
        logger.error(f"Feature preparation failed for the following dataset(s): {failed_names}")
        raise typer.Exit(code=1)

    logger.success("All requested feature artifacts generated successfully.")


if __name__ == "__main__":
    for _func in [
        _process_single_dataset,
        main,
    ]:
        _func.__module__ = MODULE_NAME

    app()
