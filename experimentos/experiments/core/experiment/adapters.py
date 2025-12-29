"""Adapter to integrate ExperimentPipeline with the training pipeline.

This module provides adapters that wrap the ExperimentPipeline to work
with the existing training infrastructure while maintaining clean separation.
"""

from pathlib import Path

from experiments.core.data import Dataset
from experiments.core.experiment.pipeline import (
    ExperimentPipeline,
    ExperimentPipelineConfig,
    ExperimentPipelineFactory,
)
from experiments.core.experiment.protocols import (
    DataPaths,
    ExperimentContext,
    ExperimentIdentity,
    TrainingConfig,
)
from experiments.core.modeling.schema import ExperimentConfig
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.model_versioning import ModelVersioningService


def create_experiment_runner(
    experiment_pipeline: ExperimentPipeline,
):
    """Create an experiment runner function from an ExperimentPipeline.

    This adapter wraps an ExperimentPipeline instance and returns a callable
    that matches the ExperimentRunner protocol signature expected by the
    training executors.

    Args:
        experiment_pipeline: The experiment pipeline to wrap.

    Returns:
        A callable that can be used as an ExperimentRunner.
    """

    def runner(
        cfg: ExperimentConfig,
        dataset_val: str,
        X_mmap_path: str,
        y_mmap_path: str,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        checkpoint_path: Path,
        model_versioning_service: ModelVersioningService | None = None,
    ) -> str | None:
        """Run a single experiment task.

        This function adapts the ExperimentPipeline.run() interface to the
        legacy ExperimentRunner signature.
        """
        # Resolve dataset
        try:
            dataset = Dataset(dataset_val)
        except ValueError:
            dataset = Dataset.from_id(str(dataset_val))

        # Create experiment context with focused dataclasses
        identity = ExperimentIdentity(
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            seed=seed,
        )

        data_paths = DataPaths(
            X_path=X_mmap_path,
            y_path=y_mmap_path,
        )

        training_config = TrainingConfig(
            cv_folds=cfg.cv_folds,
            cost_grids=cfg.cost_grids,
        )

        context = ExperimentContext(
            identity=identity,
            data=data_paths,
            config=training_config,
            checkpoint_uri=str(checkpoint_path),
            discard_checkpoints=cfg.discard_checkpoints,
        )

        # Run the experiment
        result = experiment_pipeline.run(context)

        return result.task_id

    return runner


class ExperimentRunnerFactory:
    """Factory for creating experiment runners with injected dependencies.

    This factory creates experiment runners that use the ExperimentPipeline
    architecture while maintaining compatibility with the training pipeline.
    """

    def __init__(
        self,
        pipeline_config: ExperimentPipelineConfig | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            pipeline_config: Configuration for experiment pipelines.
        """
        self._pipeline_config = pipeline_config or ExperimentPipelineConfig()

    def create_runner(
        self,
        model_versioning_service: ModelVersioningService | None = None,
    ):
        """Create an experiment runner with the given versioning service.

        Args:
            model_versioning_service: Optional service for model versioning.

        Returns:
            A callable that can be used as an ExperimentRunner.
        """
        factory = ExperimentPipelineFactory(
            model_versioning_service=model_versioning_service,
        )
        pipeline = factory.create_default_pipeline(self._pipeline_config)
        return create_experiment_runner(pipeline)

    def create_runner_with_pipeline(
        self,
        experiment_pipeline: ExperimentPipeline,
    ):
        """Create an experiment runner from an existing pipeline.

        Args:
            experiment_pipeline: The experiment pipeline to wrap.

        Returns:
            A callable that can be used as an ExperimentRunner.
        """
        return create_experiment_runner(experiment_pipeline)


__all__ = [
    "create_experiment_runner",
    "ExperimentRunnerFactory",
]
