"""CLI for training models using dependency injection pipeline."""

import random
import sys
from typing import Optional

from loguru import logger
import typer
from typing_extensions import Annotated

from experiments.containers import container
from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training import (
    ExperimentTask,
    TrainingPipelineConfig,
)
from experiments.utils.git_state import GitStateTracker

MODULE_NAME = "experiments.cli.train"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


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
    # Resolve dependencies from container
    training_factory = container.training_pipeline_factory()
    resource_calculator = container.resource_calculator()
    experiment_settings = container.settings().experiment

    tracker = GitStateTracker("train_cli")
    changed, previous_commit, current_commit = tracker.has_new_commit()

    try:
        # Git state tracking stays in CLI layer
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

        # Create pipeline configuration
        config = TrainingPipelineConfig(
            cv_folds=experiment_settings.cv_folds,
            cost_grids=experiment_settings.cost_grids,
            num_seeds=experiment_settings.num_seeds,
            discard_checkpoints=discard_checkpoints,
        )

        datasets = [dataset] if dataset is not None else list(Dataset)
        excluded_models = set(exclude_models or [])

        if len(excluded_models) == len(ModelType):
            logger.warning("All model types were excluded. No experiments to run.")
            return

        # Create pipeline and run
        n_jobs = jobs if jobs is not None else -1
        pipeline = training_factory.create_parallel_pipeline(config, n_jobs=n_jobs)

        # Define job computation function if no fixed jobs
        compute_jobs_fn = None
        if jobs is None:
            compute_jobs_fn = resource_calculator.compute_safe_jobs

        pipeline.run_all(datasets, excluded_models, compute_jobs_fn)

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
    # Resolve dependencies from container
    training_factory = container.training_pipeline_factory()
    experiment_settings = container.settings().experiment

    config = TrainingPipelineConfig(
        cv_folds=experiment_settings.cv_folds,
        cost_grids=experiment_settings.cost_grids,
        num_seeds=experiment_settings.num_seeds,
    )
    pipeline = training_factory.create_parallel_pipeline(config)

    datasets = [dataset] if dataset is not None else list(Dataset)

    for ds in datasets:
        pipeline.consolidate(ds)


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
    # Resolve dependencies from container
    training_factory = container.training_pipeline_factory()
    experiment_settings = container.settings().experiment

    config = TrainingPipelineConfig(
        cv_folds=experiment_settings.cv_folds,
        cost_grids=experiment_settings.cost_grids,
        num_seeds=experiment_settings.num_seeds,
    )
    pipeline = training_factory.create_sequential_pipeline(config)

    task = ExperimentTask(
        dataset=dataset,
        model_type=model_type,
        technique=technique,
        seed=seed,
    )

    pipeline.run_single(task)


if __name__ == "__main__":
    app()
