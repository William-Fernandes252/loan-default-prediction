from typing import Annotated

import typer

from experiments.containers import container
from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.services.training_executor import TrainModelParams

app = typer.Typer()


@app.command(name="train")
def train_model(
    dataset: Annotated[
        Dataset,
        typer.Argument(
            help="Dataset to use for training the model.",
        ),
    ],
    model_type: Annotated[ModelType, typer.Argument(help="Type of model to train.")],
    technique: Annotated[Technique, typer.Argument(help="Technique to use for the model.")],
    use_gpu: Annotated[
        bool | None,
        typer.Option("-g", "--use-gpu", help="Whether to use GPU for training."),
    ] = None,
    n_jobs: Annotated[
        int | None,
        typer.Option("-j", "--n-jobs", help="Number of parallel jobs for training."),
    ] = None,
):
    """Train a model using the specified dataset, model type, and technique."""
    model_versioner = container.model_versioner()
    resource_settings = container.settings().resources
    _, version = model_versioner.train_new_version(
        TrainModelParams(
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            use_gpu=use_gpu if use_gpu is not None else resource_settings.use_gpu,
            n_jobs=n_jobs if n_jobs is not None else resource_settings.n_jobs,
        )
    )
    typer.secho(f"Trained model version: {version.id}", fg=typer.colors.GREEN)
