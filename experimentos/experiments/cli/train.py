"""CLI for training models using dependency injection pipeline."""

import random
import sys
from typing import Optional

import typer
from typing_extensions import Annotated

from experiments.context import Context
from experiments.core.data import Dataset
from experiments.core.experiment import (
    ExperimentPipelineConfig,
    ExperimentPipelineFactory,
    create_experiment_runner,
)
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training import (
    ExperimentTask,
    TrainingPipelineConfig,
    TrainingPipelineFactory,
)
from experiments.services.data_manager import ExperimentDataManager
from experiments.utils.git_state import GitStateTracker


def _create_pipeline_factory(ctx: Context) -> TrainingPipelineFactory:
    """Create a training pipeline factory with the given context.

    The factory uses the ExperimentPipeline architecture for running
    individual experiments, wrapped in an adapter for compatibility
    with the training pipeline.

    Args:
        ctx: The application context.

    Returns:
        A configured TrainingPipelineFactory.
    """
    data_manager = ExperimentDataManager(ctx)

    # Create experiment pipeline factory (for individual experiments)
    experiment_factory = ExperimentPipelineFactory()
    experiment_pipeline = experiment_factory.create_default_pipeline(ExperimentPipelineConfig())

    # Wrap the experiment pipeline in an adapter
    experiment_runner = create_experiment_runner(experiment_pipeline)

    return TrainingPipelineFactory(
        data_provider=data_manager,
        consolidation_provider=ctx,
        versioning_provider=ctx,
        experiment_runner=experiment_runner,
    )


def _create_pipeline_config(ctx: Context) -> TrainingPipelineConfig:
    """Create a pipeline configuration from the context.

    Args:
        ctx: The application context.

    Returns:
        A configured TrainingPipelineConfig.
    """
    return TrainingPipelineConfig(
        cv_folds=ctx.cfg.cv_folds,
        cost_grids=ctx.cfg.cost_grids,
        num_seeds=ctx.cfg.num_seeds,
        discard_checkpoints=ctx.discard_checkpoints,
    )


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

        # Initialize context and pipeline
        ctx = Context(discard_checkpoints=discard_checkpoints)
        factory = _create_pipeline_factory(ctx)
        config = _create_pipeline_config(ctx)

        datasets = [dataset] if dataset is not None else list(Dataset)
        excluded_models = set(exclude_models or [])

        if len(excluded_models) == len(ModelType):
            ctx.logger.warning("All model types were excluded. No experiments to run.")
            return

        # Create pipeline and run
        n_jobs = jobs if jobs is not None else -1
        pipeline = factory.create_parallel_pipeline(config, n_jobs=n_jobs)

        # Define job computation function if no fixed jobs
        compute_jobs_fn = None
        if jobs is None:
            compute_jobs_fn = ctx.compute_safe_jobs

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
    ctx = Context()
    factory = _create_pipeline_factory(ctx)
    config = _create_pipeline_config(ctx)
    pipeline = factory.create_parallel_pipeline(config)

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
    ctx = Context()
    factory = _create_pipeline_factory(ctx)
    config = _create_pipeline_config(ctx)
    pipeline = factory.create_sequential_pipeline(config)

    task = ExperimentTask(
        dataset=dataset,
        model_type=model_type,
        technique=technique,
        seed=seed,
    )

    pipeline.run_single(task)


if __name__ == "__main__":
    app()
