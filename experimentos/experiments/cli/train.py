"""CLI for training models."""

from contextlib import contextmanager
import gc
import os
import random
import sys
import tempfile
from typing import Optional, cast

import joblib
from joblib import Parallel, delayed
import pandas as pd
import polars as pl
import typer
from typing_extensions import Annotated

from experiments.context import Context
from experiments.core.data import Dataset
from experiments.core.modeling import ModelType, Technique, run_experiment_task
from experiments.core.modeling.schema import ExperimentConfig
from experiments.utils.git_state import GitStateTracker

MODULE_NAME = "experiments.cli.train"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def _artifacts_exist(ctx: Context, dataset: Dataset) -> bool:
    paths = ctx.get_feature_paths(dataset.id)
    return paths["X"].exists() and paths["y"].exists()


def _consolidate_results(ctx: Context, dataset: Dataset):
    """Combines all checkpoint files into a single results file."""
    sample_ckpt = ctx.get_checkpoint_path(dataset.id, "x", "y", 0)
    ckpt_dir = sample_ckpt.parent

    all_files = list(ckpt_dir.glob("*.parquet"))

    if all_files:
        ctx.logger.info(f"Consolidating {len(all_files)} results for {dataset.display_name}...")
        # Read each checkpoint file and concatenate into a single DataFrame
        frames = []
        for fp in all_files:
            try:
                frames.append(pd.read_parquet(fp))
            except Exception:
                ctx.logger.warning(f"Failed to read checkpoint file {fp}, skipping.")

        if frames:
            df_final = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
            final_output = ctx.get_consolidated_results_path(dataset.id)

            # Ensure results directory exists
            final_output.parent.mkdir(parents=True, exist_ok=True)

            df_final.to_parquet(final_output)
            ctx.logger.success(f"Saved consolidated results to {final_output}")
        else:
            ctx.logger.warning(f"No readable checkpoint files found for {dataset.display_name}")
    else:
        ctx.logger.warning(f"No results found for {dataset.display_name}")


@contextmanager
def prepare_data_context(ctx: Context, dataset: Dataset):
    paths = ctx.get_feature_paths(dataset.id)
    x_path, y_path = paths["X"], paths["y"]

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Data missing for {dataset.display_name}")

    ctx.logger.info(f"Loading data for {dataset.display_name}...")

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

        # Load back briefly to check
        _ = joblib.load(X_mmap_path, mmap_mode="r")

        # Free the original RAM
        del X_df, y_df
        gc.collect()

        yield X_mmap_path, y_mmap_path


def run_dataset_experiments(
    ctx: Context,
    dataset: Dataset,
    jobs: int,
    excluded_models: set[ModelType] | None = None,
):
    """
    Prepares data and launches parallel experiment tasks for a dataset.
    """
    try:
        with prepare_data_context(ctx, dataset) as (X_mmap_path, y_mmap_path):
            # Generate Task List
            tasks = []
            excluded_models_set = excluded_models or set()
            available_models = [m for m in ModelType if m not in excluded_models_set]
            if not available_models:
                ctx.logger.warning(
                    f"No eligible models left to run for {dataset.display_name} after exclusions."
                )
                return

            cfg = ExperimentConfig(
                cv_folds=ctx.cfg.cv_folds,
                cost_grids=ctx.cfg.cost_grids,
                discard_checkpoints=ctx.discard_checkpoints,
            )

            # Access settings via Context
            for seed in range(ctx.cfg.num_seeds):
                for model_type in available_models:
                    for technique in Technique:
                        # Skip invalid combinations
                        if technique == Technique.CS_SVM and model_type != ModelType.SVM:
                            continue

                        checkpoint_path = ctx.get_checkpoint_path(
                            dataset.id, model_type.id, technique.id, seed
                        )
                        svc = ctx.get_model_versioning_service(dataset.id, model_type, technique)

                        tasks.append(
                            (
                                cfg,
                                dataset.id,
                                X_mmap_path,
                                y_mmap_path,
                                model_type,
                                technique,
                                seed,
                                checkpoint_path,
                                svc,
                            )
                        )

            ctx.logger.info(
                f"Dataset {dataset.display_name}: Launching {len(tasks)} tasks with {jobs} workers..."
            )

            # Execute Parallel
            Parallel(
                n_jobs=jobs,
                verbose=5,
                pre_dispatch="2*n_jobs",
            )(delayed(run_experiment_task)(*t) for t in tasks)

        # Consolidate results
        _consolidate_results(ctx, dataset)

    except FileNotFoundError:
        ctx.logger.error(f"Data missing for {dataset.display_name}")


@app.command("experiment")
def run_experiment(
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
            help="Number of parallel jobs. Defaults to safe number based on RAM.",
        ),
    ] = None,
    discard_checkpoints: Annotated[
        bool,
        typer.Option(
            "--discard-checkpoints",
            "-d",
            help="Delete checkpoint files so every task is rerun from scratch.",
            is_flag=True,
        ),
    ] = False,
    exclude_models: Annotated[
        list[ModelType] | None,
        typer.Option(
            "--exclude-model",
            "-x",
            help="Exclude one or more model types (use multiple flags).",
            case_sensitive=False,
        ),
    ] = None,
):
    """Runs the training experiments."""
    tracker = GitStateTracker("train_cli")
    changed, previous_commit, current_commit = tracker.has_new_commit()

    try:
        if not discard_checkpoints and changed:
            prev_short = (previous_commit or "none")[:7]
            curr_short = (current_commit or "unknown")[:7]
            prompt = (
                "New commits detected since the last run "
                f"({prev_short} -> {curr_short}).\n"
                "Do you want to discard checkpoints to rerun all experiments?"
            )
            if typer.confirm(prompt, default=True):
                discard_checkpoints = True

        # Initialize Context
        ctx = Context(discard_checkpoints=discard_checkpoints)

        datasets = [dataset] if dataset is not None else list(Dataset)
        excluded_models = set(exclude_models or [])

        if len(excluded_models) == len(ModelType):
            ctx.logger.warning("All model types were excluded. No experiments to run.")
            return

        for ds in datasets:
            if not _artifacts_exist(ctx, ds):
                ctx.logger.warning(f"Artifacts not found for {ds.display_name}. Skipping.")
                continue

            gc.collect()

            # Calculate jobs using Context helper
            if jobs is None:
                raw_path = ctx.get_raw_data_path(ds.id)
                if raw_path.exists():
                    size_gb = raw_path.stat().st_size / (1024**3)
                else:
                    size_gb = 1.0  # Default fallback

                n_jobs = ctx.compute_safe_jobs(size_gb)
            else:
                n_jobs = jobs

            run_dataset_experiments(ctx, ds, n_jobs, excluded_models)
    finally:
        tracker.record_current_commit()


@app.command("consolidate")
def consolidate(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help="Dataset to consolidate. If omitted, consolidates all datasets.",
        ),
    ] = None,
):
    """Consolidates checkpoint parquet files into the final results artifact."""

    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    for ds in datasets:
        ctx.logger.info(f"Consolidating results for {ds.display_name}...")
        _consolidate_results(ctx, ds)


@app.command("model")
def train_model(
    dataset: Annotated[
        Dataset,
        typer.Argument(
            help="Dataset to process.",
        ),
    ],
    model_type: Annotated[
        ModelType,
        typer.Argument(
            help="Model type to train.",
        ),
    ],
    technique: Annotated[
        Technique,
        typer.Argument(
            help="Technique to use for handling class imbalance.",
        ),
    ],
    seed: Annotated[
        int,
        typer.Argument(
            help="Random seed for reproducibility.",
            lazy=True,
            default_factory=lambda: random.randint(0, 1_000_000),
        ),
    ],
):
    """Runs a single model training task."""
    ctx = Context()
    try:
        with prepare_data_context(ctx, dataset) as (X_mmap_path, y_mmap_path):
            cfg = ExperimentConfig(
                cv_folds=ctx.cfg.cv_folds,
                cost_grids=ctx.cfg.cost_grids,
                discard_checkpoints=ctx.discard_checkpoints,
            )
            checkpoint_path = ctx.get_checkpoint_path(
                dataset.id, model_type.id, technique.id, seed
            )
            svc = ctx.get_model_versioning_service(dataset.id, model_type, technique)

            run_experiment_task(
                cfg,
                dataset.id,
                X_mmap_path,
                y_mmap_path,
                model_type,
                technique,
                seed,
                checkpoint_path,
                svc,
            )
    except FileNotFoundError:
        ctx.logger.error(f"Data missing for {dataset.display_name}")


if __name__ == "__main__":
    app()
