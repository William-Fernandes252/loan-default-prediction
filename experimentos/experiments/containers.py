"""Dependency injection container for the experiments application.

This module defines the Container class which manages all application
dependencies using dependency-injector. It provides centralized access
to services through the container instance.
"""

from dependency_injector import containers, providers
from loguru import logger

from experiments.core.data import DataProcessingPipelineFactory
from experiments.core.experiment import (
    ExperimentPipelineConfig,
    ExperimentPipelineFactory,
    create_experiment_runner,
)
from experiments.core.training import TrainingPipelineConfig, TrainingPipelineFactory
from experiments.services.data_manager import ExperimentDataManager
from experiments.services.model_versioning import ModelVersioningServiceFactory
from experiments.services.path_manager import PathManager
from experiments.services.resource_calculator import ResourceCalculator
from experiments.settings import ExperimentsSettings


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for the experiments application.

    This container manages all application services and their dependencies.
    Services are accessed via the container singleton instance.
    """

    # Root settings - loaded from environment/.env
    settings = providers.Singleton(ExperimentsSettings)

    # Logger - use loguru global logger
    log = providers.Object(logger)

    # --- Core Services ---

    path_manager = providers.Singleton(
        PathManager,
        settings=settings.provided.paths,
    )

    resource_calculator = providers.Singleton(
        ResourceCalculator,
        safety_factor=settings.provided.resources.safety_factor,
    )

    model_versioning_factory = providers.Singleton(
        ModelVersioningServiceFactory,
        models_dir=path_manager.provided.models_dir,
    )

    # --- Data Services ---

    data_manager = providers.Factory(
        ExperimentDataManager,
        path_manager=path_manager,
    )

    data_processing_factory = providers.Factory(
        DataProcessingPipelineFactory,
        path_provider=path_manager,
        use_gpu=settings.provided.resources.use_gpu,
    )

    # --- Experiment Pipeline ---

    experiment_pipeline_config = providers.Factory(ExperimentPipelineConfig)

    experiment_pipeline_factory = providers.Singleton(ExperimentPipelineFactory)

    experiment_pipeline = providers.Factory(
        experiment_pipeline_factory.provided.create_default_pipeline,
        config=experiment_pipeline_config,
    )

    experiment_runner = providers.Factory(
        create_experiment_runner,
        pipeline=experiment_pipeline,
    )

    # --- Training Pipeline ---

    training_pipeline_config = providers.Factory(
        TrainingPipelineConfig,
        cv_folds=settings.provided.experiment.cv_folds,
        cost_grids=settings.provided.experiment.cost_grids,
        num_seeds=settings.provided.experiment.num_seeds,
        discard_checkpoints=providers.Object(False),  # Can be overridden
    )

    training_pipeline_factory = providers.Factory(
        TrainingPipelineFactory,
        data_provider=data_manager,
        consolidation_provider=path_manager,
        versioning_provider=model_versioning_factory,
        experiment_runner=experiment_runner,
    )


def create_container() -> Container:
    """Create and initialize the DI container.

    Returns:
        Initialized Container instance.
    """
    container = Container()
    return container


# Global container instance
container = create_container()
