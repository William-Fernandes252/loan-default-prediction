from pathlib import Path
import sys

from loguru import logger
import numpy as np
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.config import PROCESSED_DATA_DIR, Dataset
from experiments.dataset import get_processed_path
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.features"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()

_BASE_PATH = PROCESSED_DATA_DIR


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
        prompt_fn=lambda ds: (f"Processed features for '{ds.value}' already exist. Overwrite?"),
        force=force,
        on_skip=lambda ds: logger.info(f"Skipping dataset {ds} per user choice."),
    )


def _ensure_artifact_directories(artifacts: dict[str, Path]) -> None:
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _process_single_dataset(dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    try:
        logger.info(f"Preparing features (X/y) for: {dataset.value}")

        input_path = get_processed_path(dataset)
        if not input_path.exists():
            msg = f"File not found: {input_path}. Run 'dataset.py' first."
            raise FileNotFoundError(msg)

        # 1. Load Intermediate Data
        logger.info(f"Loading data from {input_path}")
        df = pl.read_parquet(input_path, use_pyarrow=True)

        # 2. Split X and y
        if "target" not in df.columns:
            raise ValueError(f"Column 'target' not found in {dataset.value}")

        target_series = df.get_column("target")
        feature_df = df.drop("target")

        # 3. Sanity Check and Conversion
        # Ensure that there are no infinite values that would break Scikit-Learn
        # (dataset.py should already handle this, but we ensure it before saving)
        X_pd = feature_df.to_pandas().replace([np.inf, -np.inf], np.nan)
        y_pd = target_series.to_pandas()

        # Convert back to Polars to save efficiently in Parquet
        X_final = pl.from_pandas(X_pd)
        y_final = pl.from_pandas(y_pd.to_frame(name="target"))

        # 4. Save Artifacts
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

    for ds in datasets:
        _process_single_dataset(ds)


if __name__ == "__main__":
    app()
