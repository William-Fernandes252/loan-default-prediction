from typing import Annotated

from loguru import logger
import typer

from experiments.containers import container
from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType
from experiments.services.experiment_executor import ExperimentConfig
from experiments.services.experiment_params_resolver import (
    ResolutionContext,
    ResolutionError,
    ResolutionStatus,
    ResolverOptions,
)

app = typer.Typer()


@app.command("run")
def run(
    only_dataset: Annotated[
        Dataset | None,
        typer.Option(
            "--only-dataset",
            help="Dataset to process. If not specified, all datasets will be processed.",
        ),
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option(
            "--jobs",
            "-j",
            min=1,
            help="Number of parallel jobs. Defaults to safe number based on RAM.",
        ),
    ] = None,
    models_jobs: Annotated[
        int | None,
        typer.Option(
            "--models-jobs",
            "-m",
            min=1,
            help="Number of parallel jobs for model training. Defaults to safe number based on RAM.",
        ),
    ] = None,
    use_gpu: Annotated[
        bool | None,
        typer.Option(
            "--use-gpu",
            "-g",
            help="Utilize GPU acceleration if available during experiments.",
        ),
    ] = None,
    exclude_models: Annotated[
        list[ModelType] | None,
        typer.Option(
            "--exclude-model",
            "-x",
            help="Exclude one or more model types (use multiple flags).",
            case_sensitive=False,
        ),
    ] = None,
    execution_id: Annotated[
        str | None,
        typer.Option(
            "--execution-id",
            "-e",
            help="Execution identifier. If provided, the experiment execution that refers to it will be continued, rather than starting a new one.",
        ),
    ] = None,
    skip_resume: Annotated[
        bool,
        typer.Option(
            "--skip-resume",
            help="Skip auto-resume and start a new execution, even if incomplete executions exist.",
        ),
    ] = False,
    sequential: Annotated[
        bool | None,
        typer.Option(
            "--sequential/--no-sequential",
            "-s",
            help="Run training pipelines sequentially instead of in parallel. Reduces memory usage.",
        ),
    ] = None,
):
    """Run experiments on specified datasets and models."""
    logger.info("Starting experiment run...")

    # Get services from container
    executor = container.experiment_executor()
    resolver = container.experiment_params_resolver()

    # Build resolver options from CLI arguments
    datasets = [only_dataset] if only_dataset is not None else list(Dataset)
    options = ResolverOptions(
        datasets=datasets,
        excluded_models=exclude_models or [],
        execution_id=execution_id,
        skip_resume=skip_resume,
    )

    # Build experiment config
    config = _build_experiment_config(jobs, models_jobs, use_gpu, sequential)

    # Resolve parameters
    result = resolver.resolve_params(options, config)

    # Handle resolution errors
    if isinstance(result, ResolutionError):
        logger.error("Parameter resolution failed: {message}", message=result.message)
        if result.details:
            logger.debug("Error details: {details}", details=result.details)
        raise typer.Exit(1)

    # Extract parameters and context (guaranteed to exist after error check)
    params = result.params  # type: ignore
    context = result.context  # type: ignore
    status = context["status"]

    # Log resolution status
    _log_resolution_status(status, context)

    # Handle idempotent completion (already complete)
    if result.should_exit_early:
        logger.success(
            "Latest execution {execution_id} is complete. All combinations finished. Exiting.",
            execution_id=context["execution_id"],
        )
        raise typer.Exit(0)

    logger.debug(f"Experiment parameters: {params}")
    logger.info(
        "Executing experiment for datasets: {datasets}",
        datasets=", ".join(d.value for d in params.datasets),
    )

    # Execute the experiment
    try:
        executor.execute_experiment(params, config)
    except Exception as e:
        logger.error(f"Experiment run failed: {e}")
        raise typer.Exit(1)

    typer.echo("Experiment run completed successfully.")
    raise typer.Exit(0)


def _log_resolution_status(status: ResolutionStatus, context: ResolutionContext) -> None:
    """Log information about how parameters were resolved.

    Args:
        status: The resolution status
        context: Resolution context with metadata
    """
    datasets_str = [ds.value for ds in context["datasets"]]

    if status == ResolutionStatus.NEW_EXECUTION:
        logger.info(
            "No previous executions found for datasets: {datasets}. Starting new execution.",
            datasets=datasets_str,
        )
    elif status == ResolutionStatus.RESUMED_INCOMPLETE:
        logger.info(
            "Auto-resuming latest execution {execution_id} with {completed} completed combinations",
            execution_id=context["execution_id"],
            completed=context.get("completed_count", 0),
        )
    elif status == ResolutionStatus.SKIP_RESUME:
        logger.info(
            "Skipping auto-resume (--skip-resume flag). Starting new execution for datasets: {datasets}",
            datasets=datasets_str,
        )
    elif status == ResolutionStatus.EXPLICIT_ID_CONTINUED:
        logger.info(
            "Continuing execution {execution_id}: found {count} completed combinations",
            execution_id=context["execution_id"],
            count=context.get("completed_count", 0),
        )
    elif status == ResolutionStatus.EXPLICIT_ID_NEW:
        logger.warning(
            "Execution ID {execution_id} has no completed combinations. "
            "This will start a fresh experiment with the provided ID.",
            execution_id=context["execution_id"],
        )
    # ALREADY_COMPLETE is handled separately before this function is called


def _build_experiment_config(
    jobs: int | None,
    models_jobs: int | None,
    use_gpu: bool | None,
    sequential: bool | None,
) -> ExperimentConfig:
    """Build experiment configuration from CLI options.

    This is a simple mapping function with no business logic.

    Args:
        jobs: Number of parallel jobs for experiment execution
        models_jobs: Number of parallel jobs for model training
        use_gpu: Whether to use GPU acceleration
        sequential: Whether to run pipelines sequentially

    Returns:
        Experiment configuration dictionary
    """
    config: ExperimentConfig = {}
    if jobs is not None:
        config["n_jobs"] = jobs
    if models_jobs is not None:
        config["models_n_jobs"] = models_jobs
    if use_gpu is not None:
        config["use_gpu"] = use_gpu
    if sequential is not None:
        config["sequential"] = sequential
    return config
