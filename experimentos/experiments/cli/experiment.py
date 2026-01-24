from typing import Annotated

from loguru import logger
import typer

from experiments.containers import container
from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType
from experiments.services.experiment_executor import ExperimentParams

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
    use_gpu: Annotated[
        bool,
        typer.Option(
            "--use-gpu",
            "-g",
            help="Utilize GPU acceleration if available during experiments.",
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
    execution_id: Annotated[
        str | None,
        typer.Option(
            "--execution-id",
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

        logger.info(
            "Executing experiment for datasets: {datasets}",
            datasets=", ".join(d for d in params.datasets),
        )

        try:
            executor.execute_experiment(params)
        except Exception as e:
            logger.error(f"Experiment run failed: {e}")
            return typer.Exit(1)

        typer.echo("Experiment run completed successfully.")
        return typer.Exit(0)

    def get_experiment_params() -> ExperimentParams:
        """Construct experiment parameters based on user input."""
        datasets = filter_datasets()

        if execution_id is not None:
            logger.info(f"Continuing experiment with execution ID: {execution_id}")
            return ExperimentParams(
                datasets=datasets,
                excluded_models=exclude_models or [],
                n_jobs=get_effective_n_jobs(),
                use_gpu=get_effective_use_gpu(),
                execution_id=execution_id,
            )
        return ExperimentParams(
            datasets=datasets,
            excluded_models=exclude_models or [],
            n_jobs=get_effective_n_jobs(),
            use_gpu=get_effective_use_gpu(),
        )

    def get_effective_n_jobs() -> int:
        """Determine the effective number of parallel jobs."""
        if jobs is not None:
            return jobs
        return container.settings().resources.n_jobs

    def get_effective_use_gpu() -> bool:
        """Determine whether to use GPU acceleration."""
        return use_gpu and container.settings().resources.use_gpu

    def filter_datasets() -> list[Dataset]:
        """Filter datasets based on user input."""
        if only_dataset is not None:
            return [only_dataset]
        return list(Dataset)

    return run_experiment()
