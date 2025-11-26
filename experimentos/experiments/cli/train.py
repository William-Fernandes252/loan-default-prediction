"""CLI for training models."""

import gc
import random
import sys
from typing import Optional

from joblib import Parallel, delayed
import typer
from typing_extensions import Annotated

from experiments.context import Context
from experiments.core.data import Dataset
from experiments.core.modeling import ModelType, Technique, run_experiment_task
from experiments.core.modeling.schema import ExperimentConfig
from experiments.services.data_manager import ExperimentDataManager
from experiments.utils.git_state import GitStateTracker

MODULE_NAME = "experiments.cli.train"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


def run_dataset_experiments(
    ctx: Context,
    data_manager: ExperimentDataManager,
    dataset: Dataset,
    jobs: int,
    excluded_models: set[ModelType] | None = None,
):
    """
    Prepares data and launches parallel experiment tasks for a dataset.
    """
    try:
        with data_manager.feature_context(dataset) as (X_mmap_path, y_mmap_path):
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
        data_manager.consolidate_results(dataset)

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

        # Initialize Context and DataManager
        ctx = Context(discard_checkpoints=discard_checkpoints)
        data_manager = ExperimentDataManager(ctx)

        datasets = [dataset] if dataset is not None else list(Dataset)
        excluded_models = set(exclude_models or [])

        if len(excluded_models) == len(ModelType):
            ctx.logger.warning("All model types were excluded. No experiments to run.")
            return

        for ds in datasets:
            if not data_manager.artifacts_exist(ds):
                # Logger warning is handled inside data_manager if needed,
                # but good to check here to skip loop
                continue

            gc.collect()

            # Calculate jobs using DataManager helper
            if jobs is None:
                size_gb = data_manager.get_dataset_size_gb(ds)
                n_jobs = ctx.compute_safe_jobs(size_gb)
            else:
                n_jobs = jobs

            run_dataset_experiments(ctx, data_manager, ds, n_jobs, excluded_models)
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
    data_manager = ExperimentDataManager(ctx)
    datasets = [dataset] if dataset is not None else list(Dataset)

    for ds in datasets:
        data_manager.consolidate_results(ds)


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
    data_manager = ExperimentDataManager(ctx)

    try:
        with data_manager.feature_context(dataset) as (X_mmap_path, y_mmap_path):
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
