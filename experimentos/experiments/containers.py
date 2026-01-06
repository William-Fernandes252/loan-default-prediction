"""Dependency injection container for the experiments application.

This module defines the Container class which manages all application
dependencies using dependency-injector. It provides centralized access
to services through the container instance.
"""

from typing import Any

from dependency_injector import containers, providers
from loguru import logger

from experiments.core.data import DataProcessingPipelineFactory
from experiments.core.experiment import (
    ExperimentPipelineConfig,
    ExperimentRunnerFactory,
    create_experiment_pipeline,
)
from experiments.core.modeling.factories import DefaultEstimatorFactory
from experiments.core.training import TrainingPipelineConfig, TrainingPipelineFactory
from experiments.lib.pipelines import PipelineExecutor
from experiments.pipelines.data import (
    DataProcessingPipelineFactory as NewDataProcessingPipelineFactory,
)
from experiments.services.data_repository import DataLayout, StorageDataRepository
from experiments.services.model_versioning import ModelVersioningServiceFactory
from experiments.services.path_manager import PathManager
from experiments.services.resource_calculator import ResourceCalculator
from experiments.services.storage import LocalStorageService, StorageService
from experiments.services.storage_manager import StorageManager
from experiments.settings import ExperimentsSettings, StorageProvider
from experiments.storage import LocalStorage


def create_s3_client_from_settings(settings: ExperimentsSettings) -> Any:
    """Create an S3 client from application settings.

    Only creates the client if the provider is S3.

    Args:
        settings: Application settings.

    Returns:
        Configured boto3 S3 client, or None if not using S3.
    """
    if settings.storage.provider != StorageProvider.S3:
        return None

    from experiments.services.storage.s3 import create_s3_client

    storage = settings.storage
    return create_s3_client(
        aws_access_key_id=storage.s3_access_key_id,
        aws_secret_access_key=storage.s3_secret_access_key,
        region_name=storage.s3_region,
        endpoint_url=storage.s3_endpoint_url,
    )


def create_gcs_client_from_settings(settings: ExperimentsSettings) -> Any:
    """Create a GCS client from application settings.

    Only creates the client if the provider is GCS.

    Args:
        settings: Application settings.

    Returns:
        Configured GCS client, or None if not using GCS.
    """
    if settings.storage.provider != StorageProvider.GCS:
        return None

    from experiments.services.storage.gcs import create_gcs_client

    storage = settings.storage
    return create_gcs_client(
        project=storage.gcs_project,
        credentials_file=storage.gcs_credentials_file,
    )


def create_storage_service(
    settings: ExperimentsSettings,
    s3_client: Any = None,
    gcs_client: Any = None,
) -> StorageService:
    """Create the appropriate storage service based on settings.

    Args:
        settings: Application settings.
        s3_client: Pre-configured S3 client (injected for S3 provider).
        gcs_client: Pre-configured GCS client (injected for GCS provider).

    Returns:
        Configured storage service instance.
    """
    storage_settings = settings.storage

    if storage_settings.provider == StorageProvider.LOCAL:
        return LocalStorageService()

    elif storage_settings.provider == StorageProvider.S3:
        from experiments.services.storage import S3StorageService

        if s3_client is None:
            raise ValueError("S3 client is required for S3 storage provider")

        return S3StorageService(
            client=s3_client,
            bucket=storage_settings.s3_bucket,
            prefix=storage_settings.s3_prefix,
            cache_dir=storage_settings.cache_dir,
        )

    elif storage_settings.provider == StorageProvider.GCS:
        from experiments.services.storage import GCSStorageService

        if gcs_client is None:
            raise ValueError("GCS client is required for GCS storage provider")

        return GCSStorageService(
            client=gcs_client,
            bucket=storage_settings.gcs_bucket,
            prefix=storage_settings.gcs_prefix,
            cache_dir=storage_settings.cache_dir,
        )

    else:
        raise ValueError(f"Unknown storage provider: {storage_settings.provider}")


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for the experiments application.

    This container manages all application services and their dependencies.
    Services are accessed via the container singleton instance.
    """

    # Root settings - loaded from environment/.env
    settings = providers.Singleton(ExperimentsSettings)

    # Logger - use loguru global logger
    log = providers.Object(logger)

    # --- Cloud Clients (lazily initialized based on provider) ---

    # S3 client - only created when provider is S3
    s3_client = providers.Singleton(
        create_s3_client_from_settings,
        settings=settings,
    )

    # GCS client - only created when provider is GCS
    gcs_client = providers.Singleton(
        create_gcs_client_from_settings,
        settings=settings,
    )

    # --- Storage Layer ---

    storage_service = providers.Singleton(
        create_storage_service,
        settings=settings,
        s3_client=s3_client,
        gcs_client=gcs_client,
    )

    storage_manager = providers.Singleton(
        StorageManager,
        settings=settings.provided.paths,
        storage=storage_service,
    )

    # --- Legacy Path Manager (for backwards compatibility) ---

    path_manager = providers.Singleton(
        PathManager,
        settings=settings.provided.paths,
    )

    # --- Core Services ---

    resource_calculator = providers.Singleton(
        ResourceCalculator,
        safety_factor=settings.provided.resources.safety_factor,
    )

    model_versioning_factory = providers.Singleton(
        ModelVersioningServiceFactory,
        models_dir=path_manager.provided.models_dir,
    )

    # --- Data Processing Pipeline ---

    data_processing_factory = providers.Factory(
        DataProcessingPipelineFactory,
        storage=storage_service,
        raw_data_uri=providers.Callable(
            lambda s: StorageService.to_uri(s.paths.raw_data_dir),
            settings,
        ),
        interim_data_uri=providers.Callable(
            lambda s: StorageService.to_uri(s.paths.interim_data_dir),
            settings,
        ),
        use_gpu=settings.provided.resources.use_gpu,
    )

    # --- Estimator Factory ---

    estimator_factory = providers.Singleton(
        DefaultEstimatorFactory,
        use_gpu=settings.provided.resources.use_gpu,
    )

    # --- Experiment Pipeline ---

    experiment_pipeline_config = providers.Factory(ExperimentPipelineConfig)

    experiment_pipeline = providers.Factory(
        create_experiment_pipeline,
        storage=storage_service,
        config=experiment_pipeline_config,
        model_versioning_service_factory=model_versioning_factory,
        estimator_factory=estimator_factory,
    )

    experiment_runner_factory = providers.Factory(
        ExperimentRunnerFactory,
        storage=storage_service,
        pipeline_config=experiment_pipeline_config,
        model_versioning_service_factory=model_versioning_factory,
        estimator_factory=estimator_factory,
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
        storage=storage_service,
        data_provider=storage_manager,
        consolidation_provider=storage_manager,
        experiment_runner_factory=experiment_runner_factory,
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


class NewContainer(containers.DeclarativeContainer):
    """New dependency injection container for the experiments application.

    This container manages all application services and their dependencies.
    Services are accessed via the container singleton instance.
    """

    config = providers.Configuration(pydantic_settings=[ExperimentsSettings()])
    """Application config loaded from environment/.env."""

    logger = providers.Object(logger)
    """Logger instance using loguru."""

    _storage = providers.Singleton(LocalStorage, base_path=config.paths.project_root)
    """Storage instance.
    
    Currently uses local storage; can be extended for cloud storage. 
    """

    resource_calculator = providers.Singleton(
        ResourceCalculator,
        safety_factor=config.resources.safety_factor.as_int(),
    )
    """Resource calculator service."""

    _data_layout = providers.Singleton(DataLayout)
    """Data layout configuration."""

    data_repository = providers.Singleton(
        StorageDataRepository,
        storage=_storage,
        data_layout=_data_layout,
    )

    data_processing_pipeline_factory = providers.Singleton(
        NewDataProcessingPipelineFactory,
        data_repository=data_repository,
    )

    executor = providers.Singleton(PipelineExecutor)
