"""CLI for features and data split tasks."""

import sys

from joblib import Parallel, delayed
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.context import Context
from experiments.core.data import Dataset
from experiments.core.modeling.features import extract_features_and_target
from experiments.utils.jobs import get_jobs_from_available_cpus
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.cli.features"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _artifacts_exist(ctx: Context, dataset: Dataset) -> bool:
    paths = ctx.get_feature_paths(dataset.id)
    return any(path.exists() for path in paths.values())


def _process_single_dataset(ctx: Context, dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    """Orchestrates the feature extraction for a single dataset."""
    try:
        ctx.logger.info(f"Preparing features (X/y) for: {dataset.display_name}")

        input_path = ctx.get_interim_data_path(dataset.id)
        if not input_path.exists():
            msg = f"File not found: {input_path}. Run 'experiments.cli.data' first."
            raise FileNotFoundError(msg)

        # 1. Load Intermediate Data
        ctx.logger.info(f"Loading data from {input_path}")
        df = pl.read_parquet(input_path, use_pyarrow=True)

        # 2. Core Logic
        X_final, y_final = extract_features_and_target(df)

        # 3. Save Artifacts
        artifacts = ctx.get_feature_paths(dataset.id)
        for path in artifacts.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        ctx.logger.info(f"Saving X (shape={X_final.shape}) and y (shape={y_final.shape})...")
        X_final.write_parquet(artifacts["X"])
        y_final.write_parquet(artifacts["y"])

        ctx.logger.success(f"Processed data saved for {dataset.display_name}")
        return dataset, True, None

    except Exception as exc:  # noqa: BLE001
        ctx.logger.exception(f"Failed to process features for {dataset.display_name}: {exc}")
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
    ctx = Context()

    datasets = [dataset] if dataset is not None else list(Dataset)

    # Filter using context
    datasets = filter_items_for_processing(
        datasets,
        exists_fn=lambda ds: _artifacts_exist(ctx, ds),
        prompt_fn=lambda ds: f"Features for '{ds.display_name}' exist. Overwrite?",
        force=force,
        on_skip=lambda ds: ctx.logger.info(f"Skipping dataset {ds.display_name} per user choice."),
    )

    if not datasets:
        ctx.logger.info("No dataset selected.")
        return

    dataset_names = ", ".join(ds.display_name for ds in datasets)
    ctx.logger.info(f"Scheduling feature preparation for: {dataset_names}")

    n_jobs = get_jobs_from_available_cpus(jobs)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ctx, ds) for ds in datasets
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.display_name for ds in failed)
        ctx.logger.error(f"Feature preparation failed for: {failed_names}")
        raise typer.Exit(code=1)

    ctx.logger.success("All requested feature artifacts generated successfully.")


if __name__ == "__main__":
    for _func in [_process_single_dataset, main]:
        _func.__module__ = MODULE_NAME
    app()
