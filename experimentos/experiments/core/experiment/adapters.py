"""Adapter to integrate ExperimentPipeline with the training pipeline.

This module provides factories to create ExperimentPipeline instances
configured with the necessary services.
"""

from dataclasses import replace
from typing import Callable

from experiments.core.experiment.pipeline import (
    ExperimentPipeline,
    ExperimentPipelineConfig,
    create_experiment_pipeline,
)
from experiments.core.experiment.protocols import (
    EstimatorFactory,
    ExperimentContext,
    ExperimentResult,
)
from experiments.services.model_versioning import ModelVersioningServiceFactory
from experiments.services.storage import StorageService


class ExperimentRunnerFactory:
    """Factory for creating experiment runners with injected dependencies.

    This factory creates experiment runners that use the ExperimentPipeline
    architecture.
    """

    def __init__(
        self,
        storage: StorageService,
        pipeline_config: ExperimentPipelineConfig | None = None,
        model_versioning_service_factory: ModelVersioningServiceFactory | None = None,
        estimator_factory: EstimatorFactory | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            storage: The storage service to use for the pipeline.
            pipeline_config: Configuration for experiment pipelines.
            model_versioning_service_factory: Factory for model versioning services.
            estimator_factory: Factory for creating estimators.
        """
        self._storage = storage
        self._pipeline_config = pipeline_config or ExperimentPipelineConfig()
        self._model_versioning_service_factory = model_versioning_service_factory
        self._estimator_factory = estimator_factory

    def __call__(
        self, n_jobs_inner: int | None = None
    ) -> Callable[[ExperimentContext], ExperimentResult]:
        """Create an experiment runner with specific parallelism.

        Args:
            n_jobs_inner: Number of inner jobs for the pipeline.

        Returns:
            A callable that executes the pipeline for a given context.
        """
        pipeline = self.create_pipeline(n_jobs_inner)
        return pipeline.run

    def create_pipeline(
        self, n_jobs_inner: int | None = None
    ) -> ExperimentPipeline:
        """Create a configured ExperimentPipeline instance.

        Args:
            n_jobs_inner: Number of inner jobs for the pipeline.

        Returns:
            A configured ExperimentPipeline instance.
        """
        config = self._pipeline_config
        if n_jobs_inner is not None:
            config = replace(config, n_jobs_inner=n_jobs_inner)

        return create_experiment_pipeline(
            storage=self._storage,
            config=config,
            model_versioning_service_factory=self._model_versioning_service_factory,
            estimator_factory=self._estimator_factory,
        )


__all__ = [
    "ExperimentRunnerFactory",
]
