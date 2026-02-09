from typing import Annotated

from loguru import logger
import typer

from experiments.containers import container
from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType
from experiments.services.experiment_executor import ExperimentConfig, ExperimentParams

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
):
    """Run experiments on specified datasets and models."""
    executor = container.experiment_executor()

    def run_experiment():
        logger.info("Starting experiment run...")

        params = get_experiment_params()
        logger.debug(f"Experiment parameters: {params}")

        config = get_experiment_config()

        logger.info(
            "Executing experiment for datasets: {datasets}",
            datasets=", ".join(d for d in params.datasets),
        )

        try:
            executor.execute_experiment(params, config)
        except Exception as e:
            logger.error(f"Experiment run failed: {e}")
            return typer.Exit(1)

        typer.echo("Experiment run completed successfully.")
        return typer.Exit(0)

    def get_experiment_params() -> ExperimentParams:
        """Construct experiment parameters based on user input."""
        datasets = filter_datasets()

        # Case 1: User provided explicit execution ID → validate and use it
        if execution_id is not None:
            validate_execution_id_for_continuation(execution_id)
            return ExperimentParams(
                datasets=datasets,
                excluded_models=exclude_models or [],
                execution_id=execution_id,
            )

        # Case 2: No execution ID → auto-resume latest execution
        predictions_repo = container.model_predictions_repository()
        latest_exec_id = predictions_repo.get_latest_execution_id(datasets)

        if latest_exec_id is None:
            # No prior executions → start fresh with new ID
            logger.info(
                "No previous executions found for datasets: {datasets}. Starting new execution.",
                datasets=[ds.value for ds in datasets],
            )
            return ExperimentParams(
                datasets=datasets,
                excluded_models=exclude_models or [],
            )

        # Found latest execution → check if complete
        experiment_config = get_experiment_config()
        temp_params = ExperimentParams(
            datasets=datasets,
            excluded_models=exclude_models or [],
            execution_id=latest_exec_id,
        )

        is_complete = executor.is_execution_complete(
            latest_exec_id,
            temp_params,
            experiment_config,
        )

        if is_complete:
            # All work done → exit successfully (idempotent)
            logger.success(
                "Latest execution {execution_id} is complete. All combinations finished. Exiting.",
                execution_id=latest_exec_id,
            )
            raise typer.Exit(0)

        # Incomplete execution → resume it
        completed_count = executor.get_completed_count(latest_exec_id)
        logger.info(
            "Auto-resuming latest execution {execution_id} with {completed} completed combinations",
            execution_id=latest_exec_id,
            completed=completed_count,
        )

        return ExperimentParams(
            datasets=datasets,
            excluded_models=exclude_models or [],
            execution_id=latest_exec_id,
        )

    def get_experiment_config() -> ExperimentConfig:
        """Construct experiment configuration based on user input."""
        config: ExperimentConfig = {}
        if jobs is not None:
            config["n_jobs"] = jobs
        if models_jobs is not None:
            config["models_n_jobs"] = models_jobs
        if use_gpu is not None:
            config["use_gpu"] = use_gpu
        return config

    def validate_execution_id_for_continuation(exec_id: str) -> None:
        """Validate that the execution ID has prior work to continue."""
        completed_count = executor.get_completed_count(exec_id)

        if completed_count > 0:
            logger.info(
                "Continuing execution {execution_id}: found {count} completed combinations",
                execution_id=exec_id,
                count=completed_count,
            )
        else:
            logger.warning(
                "Execution ID {execution_id} has no completed combinations. "
                "This will start a fresh experiment with the provided ID.",
                execution_id=exec_id,
            )

    def filter_datasets() -> list[Dataset]:
        """Filter datasets based on user input."""
        if only_dataset is not None:
            return [only_dataset]
        return list(Dataset)

    return run_experiment()
