import gc
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Tuple

import joblib
from joblib import Parallel, delayed
from loguru import logger
import pandas as pd
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.config import NUM_SEEDS, PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from experiments.core.data import Dataset
from experiments.core.modeling import ModelType, Technique, run_experiment_task
from experiments.utils.jobs import get_safe_jobs

# --- Environment Configuration ---
# Limit thread usage per worker to prevent thread explosion in parallel execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

MODULE_NAME = "experiments.cli.train"

if __name__ == "__main__":
    # Fix for joblib pickling when running as __main__
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _get_training_artifact_paths(dataset: Dataset) -> Tuple[Path, Path]:
    x_path = PROCESSED_DATA_DIR / f"{dataset.value}_X.parquet"
    y_path = PROCESSED_DATA_DIR / f"{dataset.value}_y.parquet"
    return x_path, y_path


def _artifacts_exist(dataset: Dataset) -> bool:
    x_path, y_path = _get_training_artifact_paths(dataset)
    return x_path.exists() and y_path.exists()


def _get_safe_jobs_for_dataset(dataset: Dataset) -> int:
    """Determines safe number of parallel jobs for the dataset based on its size."""
    size_gb = dataset.get_size_gb(RAW_DATA_DIR)
    safe_jobs = get_safe_jobs(size_gb)
    return safe_jobs


def _consolidate_results(dataset: Dataset):
    """Combines all checkpoint files into a single results file."""
    ckpt_dir = RESULTS_DIR / "checkpoints" / dataset.value
    all_files = list(ckpt_dir.glob("*.parquet"))

    if all_files:
        logger.info(f"Consolidating {len(all_files)} results for {dataset.value}...")
        df_final = pd.read_parquet(ckpt_dir)
        final_output = RESULTS_DIR / f"{dataset.value}_results.parquet"
        df_final.to_parquet(final_output)
        logger.success(f"Saved consolidated results to {final_output}")
    else:
        logger.warning(f"No results found for {dataset.value}")


def run_dataset_experiments(dataset: Dataset, jobs: int):
    """
    Prepares data and launches parallel experiment tasks for a dataset.
    """
    x_path, y_path = _get_training_artifact_paths(dataset)
    if not x_path.exists() or not y_path.exists():
        logger.error(f"Data missing for {dataset}")
        return

    logger.info(f"Loading data for {dataset.value}...")

    # Load original data
    X_df = pl.read_parquet(x_path).to_pandas()
    y_df = pl.read_parquet(y_path).to_pandas().iloc[:, 0]

    # Create a temporary directory for memory mapping
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create memmap files
        X_mmap_path = os.path.join(temp_dir, "X.mmap")
        y_mmap_path = os.path.join(temp_dir, "y.mmap")

        # Dump data using joblib (optimized for numpy)
        joblib.dump(X_df.to_numpy(), X_mmap_path)
        joblib.dump(y_df.to_numpy(), y_mmap_path)

        # Load back briefly to get shapes/dtypes (sanity check)
        # and to ensure the file is ready
        _ = joblib.load(X_mmap_path, mmap_mode="r")

        # Free the original RAM immediately
        del X_df, y_df
        gc.collect()

        # Generate Task List
        tasks = []
        for seed in range(NUM_SEEDS):
            for model_type in ModelType:
                for technique in Technique:
                    # Skip invalid combinations
                    if technique == Technique.CS_SVM and model_type != ModelType.SVM:
                        continue

                    tasks.append(
                        (
                            dataset.value,
                            X_mmap_path,
                            y_mmap_path,
                            model_type,
                            technique,
                            seed,
                        )
                    )

        logger.info(
            f"Dataset {dataset.value}: Launching {len(tasks)} tasks with {jobs} workers..."
        )

        # Execute Parallel using the core runner
        Parallel(
            n_jobs=jobs,
            verbose=5,
            pre_dispatch="2*n_jobs",
        )(delayed(run_experiment_task)(*t) for t in tasks)

    # Consolidate results
    _consolidate_results(dataset)


@app.command()
def main(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help="Dataset to process. If not specified, all datasets will be processed."
        ),
    ] = None,
    jobs: Annotated[
        Optional[int],
        typer.Option(
            "--jobs",
            "-j",
            min=1,
            help="Number of parallel jobs to run. If not specified, a safe number based on dataset size will be used.",
        ),
    ] = None,
):
    """
    Runs the training experiments.
    """
    datasets = [dataset] if dataset is not None else list(Dataset)

    for ds in datasets:
        if not _artifacts_exist(ds):
            logger.warning(f"Artifacts not found for {ds}. Skipping.")
            continue

        gc.collect()
        n_jobs = jobs if jobs is not None else _get_safe_jobs_for_dataset(ds)
        run_dataset_experiments(ds, n_jobs)


if __name__ == "__main__":
    app()
