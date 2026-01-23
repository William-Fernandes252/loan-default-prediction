"""Implements the default training execution operations."""

from typing import TYPE_CHECKING, Callable, Self

from pydantic import BaseModel, PositiveInt, model_validator

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ClassifierFactory, ModelType, Technique
from experiments.core.training.data import TrainingDataLoader
from experiments.core.training.splitters import DataSplitter
from experiments.core.training.trainers import ModelTrainer, TrainedModel
from experiments.lib.pipelines import PipelineExecutor
from experiments.pipelines.training.factory import TrainingPipelineFactory
from experiments.pipelines.training.pipeline import TrainingPipelineContext, TrainingPipelineState

if TYPE_CHECKING:
    from experiments.pipelines.training.factory import TrainingPipelineFactory

type SeedGenerator = Callable[[], int]
"""Generates random seeds for reproducibility."""


class TrainingParams(BaseModel):
    """Parameters for training a model."""

    dataset: Dataset
    """The dataset to train on."""

    model_type: ModelType
    """The type of the model to be trained."""

    technique: Technique
    """The technique used by the model."""

    n_jobs: PositiveInt = 1
    """The number of parallel jobs to run during training."""

    use_gpu: bool = False
    """Whether to use GPU for training."""

    @model_validator(mode="after")
    def ensure_valid_type_and_technique_combination(self) -> Self:
        """Ensures that the model type and technique combination is valid.

        Raises:
            ValueError: If the combination is invalid.
        """
        if self.model_type == ModelType.SVM and self.technique == Technique.CS_SVM:
            raise ValueError("Cost-sensitive SVM is not supported for SVM model type.")
        return self


class TrainingExecutor:
    """Service for training and testing models."""

    def __init__(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        pipeline_executor: PipelineExecutor,
        model_trainer: ModelTrainer,
        data_splitter: DataSplitter,
        training_data_loader: TrainingDataLoader,
        classifier_factory: ClassifierFactory,
        seed_generator: SeedGenerator,
    ):
        self._training_pipeline_factory = training_pipeline_factory
        self._pipeline_executor = pipeline_executor
        self._model_trainer = model_trainer
        self._data_splitter = data_splitter
        self._training_data_loader = training_data_loader
        self._classifier_factory = classifier_factory
        self._seed_generator = seed_generator

    def train_model(
        self,
        params: TrainingParams,
    ) -> TrainedModel:
        """Trains a model using the training pipeline.

        Args:
            params (TrainModelParams): Parameters for training the model.

        Returns:
            The trained model.
        """
        seed = self._seed_generator()

        pipeline = self._training_pipeline_factory.create_pipeline(
            name=self._get_pipeline_name(
                params,
                seed,
            )
        )

        context = TrainingPipelineContext(
            dataset=params.dataset,
            model_type=params.model_type,
            technique=params.technique,
            data_splitter=self._data_splitter,
            training_data_loader=self._training_data_loader,
            classifier_factory=self._classifier_factory,
            trainer=self._model_trainer,
            seed=seed,
            n_jobs=params.n_jobs,
            use_gpu=params.use_gpu,
        )

        result = self._pipeline_executor.execute(pipeline, TrainingPipelineState(), context)

        if not result.succeeded():
            raise RuntimeError(f"Training pipeline failed: {result.last_error()}")

        return result.final_state["trained_model"]

    def _get_pipeline_name(
        self,
        params: TrainingParams,
        seed: int,
    ) -> str:
        """Generates a name for the training pipeline.

        Args:
            params (TrainModelParams): The training parameters.

        Returns:
            str: The generated pipeline name.
        """
        return f"TrainingPipeline[dataset={params.dataset}, model_type={params.model_type}, technique={params.technique.value}, seed={seed}]"
