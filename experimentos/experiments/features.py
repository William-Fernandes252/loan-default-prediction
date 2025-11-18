from pathlib import Path
import sys

import joblib
from joblib import Parallel, cpu_count, delayed
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from polars import selectors as cs
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer
from typing_extensions import Annotated

from experiments.config import PROCESSED_DATA_DIR, Dataset
from experiments.dataset import get_processed_path
from experiments.utils.overwrites import filter_items_for_processing

MODULE_NAME = "experiments.features"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()

# --- Specific Pipeline Functions ---


_BASE_PATH = PROCESSED_DATA_DIR


def _get_artifact_paths(dataset: Dataset) -> dict[str, Path]:
    """Return all output artifact paths for a dataset."""

    return {
        "pipeline": _BASE_PATH / f"{dataset.value}_preprocessor.joblib",
        "X_train": _BASE_PATH / f"{dataset.value}_X_train_processed.parquet",
        "X_test": _BASE_PATH / f"{dataset.value}_X_test_processed.parquet",
        "y_train": _BASE_PATH / f"{dataset.value}_y_train.parquet",
        "y_test": _BASE_PATH / f"{dataset.value}_y_test.parquet",
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
            f"Processed feature artifacts for '{ds.value}' already exist. Overwrite them?"
        ),
        force=force,
        on_skip=lambda ds: logger.info(f"Skipping dataset {ds} per user choice."),
    )


def _ensure_artifact_directories(artifacts: dict[str, Path]) -> None:
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_features(df: pl.DataFrame) -> pl.DataFrame:
    # Sanitization of numerical infinities should happen during dataset processing
    # (data cleaning/parsing). This function is kept for backward compatibility but
    # is now a no-op here.
    return df


def _should_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    if counts.empty:
        return False
    if counts.min() < 2:
        logger.warning(
            "Skipping stratified split because at least one class has fewer than 2 samples."
        )
        return False
    return True


def get_preprocessor(df: pl.DataFrame) -> ColumnTransformer:
    """
    Creates and returns the ColumnTransformer (preprocessing pipeline)
    based on the DataFrame columns.

    This function assumes the data has already gone through Step 2's
    preprocessing (target binarization, one-hot encoding).
    """

    # Identify all numeric columns (except the 'target')
    # At this point, all features are already numeric (original or one-hot)
    numeric_features = df.select(cs.numeric().exclude("target")).columns
    logger.info(f"Identified {len(numeric_features)} numeric features for imputation and scaling.")

    # Create the preprocessing pipeline
    # 1. Impute missing values (nulls) with the median (robust to outliers)
    # 2. Scale (normalize) features to a mean of 0 and a standard deviation of 1
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Create the ColumnTransformer
    # We apply the 'numeric_transformer' to all 'numeric_features'
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",  # Discards unlisted columns (none, in this case)
    )

    return preprocessor


# --- Main Command (Typer) ---


def _process_single_dataset(dataset: Dataset) -> tuple[Dataset, bool, str | None]:
    try:
        logger.info(f"Loading features for dataset: {dataset.value}")

        input_path = get_processed_path(dataset)
        if not input_path.exists():
            msg = f"File not found: {input_path}. Run the dataset processing command first."
            raise FileNotFoundError(msg)

        logger.info(f"Loading data from {input_path}")
        df = pl.read_parquet(input_path, use_pyarrow=True)
        df = _sanitize_features(df)

        feature_df = df.drop("target")
        target_series = df.get_column("target")
        X = feature_df.to_pandas()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y = target_series.to_pandas()

        logger.info("Splitting data into training (70%) and testing (30%) with stratification...")
        stratify_labels = y if _should_stratify(y) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=stratify_labels,
        )

        logger.info("Defining the preprocessing pipeline (Imputer + Scaler)...")
        preprocessor = get_preprocessor(feature_df)

        logger.info("Fitting the preprocessing pipeline ONLY on the training data...")
        X_train_processed = preprocessor.fit_transform(X_train)

        logger.info("Applying (transform) the pipeline to the test data...")
        X_test_processed = preprocessor.transform(X_test)

        feature_names = preprocessor.get_feature_names_out()

        artifacts = _get_artifact_paths(dataset)
        _ensure_artifact_directories(artifacts)

        joblib.dump(preprocessor, artifacts["pipeline"])
        logger.success(f"Preprocessing pipeline saved to: {artifacts['pipeline']}")

        pl.DataFrame(X_train_processed, schema=feature_names.tolist()).write_parquet(
            artifacts["X_train"]
        )
        pl.DataFrame(X_test_processed, schema=feature_names.tolist()).write_parquet(
            artifacts["X_test"]
        )
        pl.DataFrame({"target": y_train.reset_index(drop=True).tolist()}).write_parquet(
            artifacts["y_train"]
        )
        pl.DataFrame({"target": y_test.reset_index(drop=True).tolist()}).write_parquet(
            artifacts["y_test"]
        )

        logger.success("Processed train/test data saved in 'data/processed/'")
        return dataset, True, None
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to process features for {dataset}: {exc}")
        return dataset, False, str(exc)


@app.command()
def main(
    dataset: Annotated[
        Dataset | None,
        typer.Argument(
            help=(
                "Dataset whose features should be processed. "
                "When omitted, all datasets are processed in parallel."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing feature artifacts without prompting.",
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
    """Splits data, fits preprocessing pipelines, and saves artifacts in parallel."""

    datasets = [dataset] if dataset is not None else list(Dataset)
    datasets = _filter_datasets_for_processing(datasets, force=force)

    if not datasets:
        logger.info("No datasets selected for feature processing. Exiting.")
        return

    dataset_names = ", ".join(ds.value for ds in datasets)
    logger.info(f"Scheduling feature processing for {len(datasets)} dataset(s): {dataset_names}")

    available_cpus = cpu_count() or 1
    requested_jobs = jobs if jobs is not None else available_cpus
    n_jobs = min(len(datasets), max(1, requested_jobs))

    if jobs is not None:
        logger.info(f"Using {n_jobs} parallel job(s) managed by joblib (user requested {jobs}).")
    else:
        logger.info(
            f"Using {n_jobs} parallel job(s) managed by joblib (detected {available_cpus} CPU(s))."
        )

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_process_single_dataset)(ds) for ds in datasets
    )

    failed = [ds for ds, success, _ in results if not success]
    if failed:
        failed_names = ", ".join(ds.value for ds in failed)
        logger.error(f"Feature processing failed for the following dataset(s): {failed_names}")
        raise typer.Exit(code=1)

    logger.success("All requested feature pipelines processed successfully.")


if __name__ == "__main__":
    for _func in [
        get_preprocessor,
        _process_single_dataset,
        main,
    ]:
        _func.__module__ = MODULE_NAME

    app()
