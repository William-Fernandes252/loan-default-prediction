"""Definition of the experiment execution service."""

import gc
from typing import Generator, TypedDict, override

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from uuid_extensions import uuid7str

from experiments.config.settings import ExperimentSettings, ResourceSettings
from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ClassifierFactory, ModelType, Technique
from experiments.core.predictions.repository import (
    ExperimentCombination,
    ModelPredictionsRepository,
)
from experiments.core.training.data import TrainingDataLoader
from experiments.core.training.splitters import DataSplitter
from experiments.core.training.trainers import ModelTrainer
from experiments.lib.pipelines import Action, IgnoreAllObserver, PipelineExecutor
from experiments.lib.pipelines.lifecycle import PipelineObserver
from experiments.pipelines.training.factory import TrainingPipelineFactory
from experiments.pipelines.training.pipeline import (
    TrainingPipeline,
    TrainingPipelineContext,
    TrainingPipelineState,
)


class ExperimentConfig(TypedDict, total=False):
    """Configuration for experiments."""

    num_seeds: int
    use_gpu: bool
    n_jobs: int
    models_n_jobs: int
    sequential: bool


class ExperimentParams(BaseModel):
    """Parameters for executing an experiment."""

    datasets: list[Dataset] = Field(
        default_factory=lambda: list(Dataset),
        json_schema_extra={"description": "List of datasets to include in the experiment."},
    )
    excluded_models: list[ModelType] = Field(
        default_factory=list,
        json_schema_extra={"description": "List of model types to exclude from the experiment."},
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
    IgnoreAllObserver[TrainingPipelineState, TrainingPipelineContext],
):
    """Observer that saves predictions when training completes."""

    def __init__(
        self, execution_id: str, predictions_repository: ModelPredictionsRepository
    ) -> None:
        self._execution_id = execution_id
        self._predictions_repository = predictions_repository

    @override
    def on_pipeline_finish(self, pipeline, result):
        with logger.contextualize(
            model_type=result.context.model_type,
            technique=result.context.technique,
            seed=result.context.seed,
        ):
            if "predictions" not in result.final_state:
                if not result.succeeded():
                    logger.warning("Pipeline failed; skipping prediction save.")
                else:
                    logger.warning("Pipeline finished but no predictions found in state.")
                return Action.PROCEED

            try:
                self._predictions_repository.save_predictions(
                    execution_id=self._execution_id,
                    seed=result.context.seed,
                    dataset=result.context.dataset,
                    model_type=result.context.model_type,
                    technique=result.context.technique,
                    predictions=result.final_state["predictions"],
                )
            except Exception as e:
                logger.error(
                    "Failed to save predictions for dataset: {error}",
                    error=e,
                )
                return Action.PANIC
            else:
                logger.info("Saved predictions for dataset.")
                return Action.PROCEED
            finally:
                result.final_state = {}  # Clear state to free memory


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
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        self._training_pipeline_factory = training_pipeline_factory
        self._pipeline_executor = pipeline_executor
        self._model_trainer = model_trainer
        self._data_splitter = data_splitter
        self._training_data_loader = training_data_loader
        self._classifier_factory = classifier_factory
        self._predictions_repository = predictions_repository
        self._experiment_settings = experiment_settings
        self._resource_settings = resource_settings

        self._default_config: ExperimentConfig = {
            "num_seeds": self._experiment_settings.num_seeds,
            "use_gpu": self._resource_settings.use_gpu,
            "n_jobs": self._resource_settings.n_jobs,
            "models_n_jobs": self._resource_settings.models_n_jobs,
            "sequential": self._resource_settings.sequential,
        }

    def execute_experiment(
        self, params: ExperimentParams, config: ExperimentConfig | None = None
    ) -> None:
        """Execute the experiment with the given parameters."""
        config = self._merge_with_default_config(config or {})

        with logger.contextualize(
            execution_id=params.execution_id,
            num_seeds=config["num_seeds"],
            n_jobs=config["n_jobs"],
            models_n_jobs=config["models_n_jobs"],
            **{dataset.value: (dataset in params.datasets) for dataset in Dataset},
        ):
            if config.get("sequential", False):
                self._execute_experiment_sequentially(params, config)
            else:
                self._schedule_pipelines(params, config)
                self._execute_pipelines(params.execution_id, config["n_jobs"])

    def _merge_with_default_config(self, config: ExperimentConfig) -> ExperimentConfig:
        """Get the default experiment configuration from settings."""
        merged_config = self._default_config.copy()
        merged_config.update(config)
        return merged_config

    def _schedule_pipelines(self, params: ExperimentParams, config: ExperimentConfig) -> None:
        """Schedule training pipelines for all valid model/technique/seed combinations."""
        completed = self._get_completed_combinations(params.execution_id)

        scheduled_count = 0
        skipped_count = 0

        for dataset in params.datasets:
            for model_type in self._get_model_types(params):
                for technique in Technique:
                    if self._is_valid_combination(model_type, technique):
                        for seed in self._generate_seeds(config["num_seeds"]):
                            combination = ExperimentCombination(
                                dataset=dataset,
                                model_type=model_type,
                                technique=technique,
                                seed=seed,
                            )
                            if combination in completed:
                                skipped_count += 1
                                continue

                            pipeline, initial_state, context = self._create_training_pipeline(
                                dataset, model_type, technique, seed, config
                            )
                            self._pipeline_executor.schedule(pipeline, initial_state, context)
                            scheduled_count += 1

        if skipped_count > 0:
            logger.info(
                "Continuing execution: scheduled {scheduled}, skipped {skipped} completed combinations",
                scheduled=scheduled_count,
                skipped=skipped_count,
            )
        else:
            logger.info(
                "Execution: scheduled {scheduled} combinations",
                scheduled=scheduled_count,
            )

    def _execute_experiment_sequentially(
        self, params: ExperimentParams, config: ExperimentConfig
    ) -> None:
        """Execute training pipelines one at a time, freeing memory between each."""
        completed = self._get_completed_combinations(params.execution_id)

        scheduled_count = 0
        skipped_count = 0

        observers: set[PipelineObserver[TrainingPipelineState, TrainingPipelineContext]] = {
            _SavePredictionsOnTrainingCompletionObserver(
                params.execution_id, self._predictions_repository
            )
        }

        for dataset in params.datasets:
            for model_type in self._get_model_types(params):
                for technique in Technique:
                    if self._is_valid_combination(model_type, technique):
                        for seed in self._generate_seeds(config["num_seeds"]):
                            combination = ExperimentCombination(
                                dataset=dataset,
                                model_type=model_type,
                                technique=technique,
                                seed=seed,
                            )
                            if combination in completed:
                                skipped_count += 1
                                continue

                            scheduled_count += 1

                            pipeline, initial_state, context = self._create_training_pipeline(
                                dataset, model_type, technique, seed, config
                            )
                            self._pipeline_executor.schedule(pipeline, initial_state, context)
                            self._pipeline_executor.start(
                                observers=observers,
                                max_workers=config["n_jobs"],
                            )
                            self._pipeline_executor.wait()
                            self._pipeline_executor.reset()
                            gc.collect()

        if skipped_count > 0:
            logger.info(
                "Sequential execution complete: ran {scheduled}, skipped {skipped} completed combinations",
                scheduled=scheduled_count,
                skipped=skipped_count,
            )
        else:
            logger.info(
                "Sequential execution complete: ran {scheduled} combinations",
                scheduled=scheduled_count,
            )

    def _get_completed_combinations(self, execution_id: str) -> set[ExperimentCombination]:
        """Get completed combinations for the execution, with validation."""
        completed = self._predictions_repository.get_completed_combinations(execution_id)

        if not completed:
            # Check if this looks like a continuation attempt (existing execution_id passed)
            # We can't distinguish new vs continuation here, but log for visibility
            logger.debug(
                "No completed combinations found for execution.",
            )

        return completed

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
        config: ExperimentConfig,
    ) -> tuple[TrainingPipeline, TrainingPipelineState, TrainingPipelineContext]:
        """Create a training pipeline for the given parameters."""
        pipeline = self._training_pipeline_factory.create_pipeline(
            name=self._get_pipeline_name(dataset, model_type, technique, seed)
        )
        context = self._create_training_pipeline_context(
            dataset, model_type, technique, seed, config
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
        config: ExperimentConfig,
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
            use_gpu=config.get("use_gpu", False),
            n_jobs=config.get("models_n_jobs", 1),
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

    def get_completed_count(self, execution_id: str) -> int:
        """Get the number of completed combinations for an execution.

        Args:
            execution_id: The execution identifier to check.

        Returns:
            The number of completed combinations, or 0 if none exist.
        """
        return len(self._predictions_repository.get_completed_combinations(execution_id))

    def is_execution_complete(
        self,
        execution_id: str,
        params: ExperimentParams,
        config: ExperimentConfig,
    ) -> bool:
        """Check if an execution has completed all expected combinations.

        Args:
            execution_id: The execution to check.
            params: Experiment parameters (datasets, excluded models).
            config: Experiment configuration (num_seeds, etc.).

        Returns:
            True if all expected combinations have been executed, False otherwise.
        """
        completed = self._predictions_repository.get_completed_combinations(execution_id)
        completed_count = len(completed)

        # Calculate expected total using the same logic as _schedule_pipelines
        expected_count = 0
        for dataset in params.datasets:
            for model_type in self._get_model_types(params):
                for technique in Technique:
                    if self._is_valid_combination(model_type, technique):
                        expected_count += config.get(
                            "num_seeds", self._default_config["num_seeds"]
                        )

        return completed_count >= expected_count
