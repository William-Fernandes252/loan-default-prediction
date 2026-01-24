"""Definition of the experiment execution service."""

from typing import Generator, override

from pydantic import BaseModel, Field, PositiveInt, field_validator
from uuid_extensions import uuid7str

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ClassifierFactory, ModelType, Technique
from experiments.core.predictions.repository import ModelPredictionsRepository
from experiments.core.training.data import TrainingDataLoader
from experiments.core.training.splitters import DataSplitter
from experiments.core.training.trainers import ModelTrainer
from experiments.lib.pipelines import PipelineExecutor
from experiments.lib.pipelines.lifecycle import IgnoreAllObserver
from experiments.pipelines.training.factory import TrainingPipelineFactory
from experiments.pipelines.training.pipeline import (
    TrainingPipeline,
    TrainingPipelineContext,
    TrainingPipelineState,
)


class ExperimentParams(BaseModel):
    """Parameters for executing an experiment."""

    n_jobs: PositiveInt = Field(
        default=1,
        json_schema_extra={
            "description": "Number of parallel jobs to use for experiment execution."
        },
    )
    datasets: list[Dataset] = Field(
        default_factory=lambda: list(Dataset),
        json_schema_extra={"description": "List of datasets to include in the experiment."},
    )
    excluded_models: list[ModelType] = Field(
        default_factory=list,
        json_schema_extra={"description": "List of model types to exclude from the experiment."},
    )
    num_seeds: PositiveInt = Field(
        default=30,
        json_schema_extra={"description": "Number of random seeds to use for experiment tasks."},
    )
    use_gpu: bool = Field(
        default=False,
        json_schema_extra={"description": "Whether to use GPU for training."},
    )
    execution_id: str = Field(
        default_factory=uuid7str,
        json_schema_extra={"description": "Unique identifier for the experiment execution."},
    )

    @field_validator("excluded_models")
    @classmethod
    def does_not_exclude_all_models(cls, value) -> list[ModelType]:
        """Ensure that not all model types are excluded."""
        if len(value) == len(ModelType):
            raise ValueError("At least one model type must be included in the experiment.")
        return value


class _SavePredictionsOnTrainingCompletionObserver(
    IgnoreAllObserver[TrainingPipelineState, TrainingPipelineContext]
):
    """Observer that saves predictions when training completes."""

    def __init__(
        self, execution_id: str, predictions_repository: ModelPredictionsRepository
    ) -> None:
        self._execution_id = execution_id
        self._predictions_repository = predictions_repository

    @override
    def on_pipeline_finish(self, pipeline, result):
        trained_model = result.final_state["trained_model"]
        if trained_model is not None:
            predictions = trained_model.model.predict(result.final_state["data_split"].X_test)
            self._predictions_repository.save_predictions(
                execution_id=self._execution_id,
                seed=result.context.seed,
                dataset=result.context.dataset,
                model_type=result.context.model_type,
                technique=result.context.technique,
                predictions=predictions,
            )

        return super().on_pipeline_finish(pipeline, result)


class ExperimentExecutor:
    """Service for executing the experiment."""

    def __init__(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        pipeline_executor: PipelineExecutor,
        model_trainer: ModelTrainer,
        data_splitter: DataSplitter,
        training_data_loader: TrainingDataLoader,
        classifier_factory: ClassifierFactory,
        predictions_repository: ModelPredictionsRepository,
    ) -> None:
        self._training_pipeline_factory = training_pipeline_factory
        self._pipeline_executor = pipeline_executor
        self._model_trainer = model_trainer
        self._data_splitter = data_splitter
        self._training_data_loader = training_data_loader
        self._classifier_factory = classifier_factory
        self._predictions_repository = predictions_repository

    def execute_experiment(self, params: ExperimentParams) -> None:
        """Execute the experiment with the given parameters."""
        self._schedule_pipelines(params)
        self._execute_pipelines(params.execution_id, params.n_jobs)

    def _schedule_pipelines(self, params: ExperimentParams) -> None:
        """Schedule training pipelines for all valid model/technique/seed combinations."""
        for dataset in params.datasets:
            for model_type in self._get_model_types(params):
                for technique in Technique:
                    if self._is_valid_combination(model_type, technique):
                        for seed in self._generate_seeds(params.num_seeds):
                            pipeline, initial_state, context = self._create_training_pipeline(
                                dataset, model_type, technique, seed, use_gpu=params.use_gpu
                            )
                            self._pipeline_executor.schedule(pipeline, initial_state, context)

    @staticmethod
    def _is_valid_combination(model_type: ModelType, technique: Technique) -> bool:
        """Check if a model/technique combination is valid."""
        # CS_SVM is only valid for SVM models
        if technique == Technique.CS_SVM and model_type != ModelType.SVM:
            return False
        return True

    def _get_model_types(self, params: ExperimentParams) -> list[ModelType]:
        """Get the list of model types to include in the experiment."""
        return [mt for mt in ModelType if mt not in params.excluded_models]

    def _generate_seeds(self, num_seeds: int) -> Generator[int, None, None]:
        """Generate a list of random seeds."""
        for i in range(num_seeds):
            yield i + 1  # Generate seeds starting from 1 to `num_seeds`

    def _create_training_pipeline(
        self,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        use_gpu: bool,
    ) -> tuple[TrainingPipeline, TrainingPipelineState, TrainingPipelineContext]:
        """Create a training pipeline for the given parameters."""
        pipeline = self._training_pipeline_factory.create_pipeline(
            name=self._get_pipeline_name(dataset, model_type, technique, seed)
        )
        context = self._create_training_pipeline_context(
            dataset, model_type, technique, seed, use_gpu=use_gpu
        )
        initial_state = self._get_training_pipeline_initial_state()
        return pipeline, initial_state, context

    def _get_pipeline_name(
        self,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        seed: int,
    ) -> str:
        """Generate a unique name for the training pipeline."""
        return f"TrainingPipeline[dataset={dataset}, model_type={model_type}, technique={technique}, seed={seed}]"

    def _create_training_pipeline_context(
        self,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        use_gpu: bool,
    ) -> TrainingPipelineContext:
        """Create the context for a training pipeline."""
        return TrainingPipelineContext(
            trainer=self._model_trainer,
            data_splitter=self._data_splitter,
            classifier_factory=self._classifier_factory,
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            seed=seed,
            training_data_loader=self._training_data_loader,
            use_gpu=use_gpu,
            n_jobs=1,  # Each pipeline runs in its own thread
        )

    def _get_training_pipeline_initial_state(self) -> TrainingPipelineState:
        """Get the initial state for a training pipeline."""
        return TrainingPipelineState()

    def _execute_pipelines(self, execution_id: str, workers: int) -> None:
        """Execute all scheduled pipelines."""
        self._pipeline_executor.start(
            observers={
                _SavePredictionsOnTrainingCompletionObserver(
                    execution_id, self._predictions_repository
                )
            },
            max_workers=workers,
        )
        self._pipeline_executor.wait()
