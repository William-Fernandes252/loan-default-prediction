"""Dependency injection container for the experiments application.

This module defines the Container class which manages all application
dependencies using dependency-injector. It provides centralized access
to services through the container instance.
"""

from typing import Any

from dependency_injector import containers, providers
from loguru import logger

from experiments.config.logging import LoggingObserver
from experiments.config.settings import LdpSettings, StorageProvider
from experiments.lib.pipelines.execution import PipelineExecutor
from experiments.pipelines.data import (
    DataProcessingPipelineFactory as NewDataProcessingPipelineFactory,
)
from experiments.services.data_manager import DataManager
from experiments.services.data_repository import DataStorageLayout, StorageDataRepository
from experiments.services.feature_extractor import FeatureExtractorImpl
from experiments.services.grid_search_trainer import GridSearchModelTrainer
from experiments.services.resource_calculator import ResourceCalculator
from experiments.services.umbalanced_learner_factory import UnbalancedLearnerFactory
from experiments.storage import GCSStorage, LocalStorage, S3Storage, Storage


def create_s3_client_from_settings(settings: LdpSettings) -> Any:
    """Create an S3 client from application settings.

    Only creates the client if the provider is S3.

    Args:
        settings: Application settings.

    Returns:
        Configured boto3 S3 client, or None if not using S3.
    """
    if settings.storage.provider != StorageProvider.S3:
        return None

    from experiments.storage.s3 import create_s3_client

    storage = settings.storage
    return create_s3_client(
        aws_access_key_id=storage.s3_access_key_id,
        aws_secret_access_key=storage.s3_secret_access_key,
        region_name=storage.s3_region,
        endpoint_url=storage.s3_endpoint_url,
    )


def create_gcs_client_from_settings(settings: LdpSettings) -> Any:
    """Create a GCS client from application settings.

    Only creates the client if the provider is GCS.

    Args:
        settings: Application settings.

    Returns:
        Configured GCS client, or None if not using GCS.
    """
    if settings.storage.provider != StorageProvider.GCS:
        return None

    from experiments.storage.gcs import create_gcs_client

    storage = settings.storage
    return create_gcs_client(
        project=storage.gcs_project,
        credentials_file=storage.gcs_credentials_file,
    )


def create_storage(
    settings: LdpSettings,
    s3_client: Any = None,
    gcs_client: Any = None,
) -> Storage:
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
        return LocalStorage(storage_settings.base_path)

    elif storage_settings.provider == StorageProvider.S3:
        if s3_client is None or storage_settings.s3_bucket is None:
            raise ValueError("S3 client and bucket name are required for S3 storage provider")

        return S3Storage(
            s3_client=s3_client,
            bucket_name=storage_settings.s3_bucket,
            cache_dir=storage_settings.cache_dir,
        )

    elif storage_settings.provider == StorageProvider.GCS:
        if gcs_client is None or storage_settings.gcs_bucket is None:
            raise ValueError("GCS client and bucket name are required for GCS storage provider")

        return GCSStorage(
            gcs_client=gcs_client,
            bucket_name=storage_settings.gcs_bucket,
            cache_dir=storage_settings.cache_dir,
        )

    else:
        raise ValueError(f"Unknown storage provider: {storage_settings.provider}")


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for the experiments application.

    This container manages all application services and their dependencies.
    Services are accessed via the container singleton instance.
    """

    settings = providers.Singleton(LdpSettings)
    """Application config loaded from environment/.env."""

    logger = providers.Object(logger)
    """Global logger instance."""

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

    _storage = providers.Singleton(
        create_storage,
        settings=settings,
        s3_client=s3_client,
        gcs_client=gcs_client,
    )
    """Storage service based on configured provider."""

    # --- Logging ---

    logger = providers.Object(logger)
    """Logger instance using loguru."""

    # --- Core Services ---

    resource_calculator = providers.Singleton(
        ResourceCalculator,
        safety_factor=settings.provided.resources.safety_factor,
    )
    """Resource calculator service."""

    classifier_factory = providers.Singleton(
        UnbalancedLearnerFactory,
        use_gpu=settings.provided.resources.use_gpu,
    )
    """Factory for creating unbalanced learner classifiers."""

    model_trainer = providers.Singleton(
        GridSearchModelTrainer,
        cv_folds=settings.provided.experiment.cv_folds,
        cost_grids=settings.provided.experiment.cost_grids,
    )
    """Model trainer service for training and hyperparameter tuning."""

    data_repository = providers.Singleton(
        StorageDataRepository,
        storage=_storage,
        data_layout=providers.Singleton(DataStorageLayout),
    )
    """Data repository service for dataset storage and retrieval."""

    resource_calculator = providers.Singleton(
        ResourceCalculator,
        safety_factor=settings.provided.resources.safety_factor,
    )
    """Resource calculator service."""

    _data_layout = providers.Singleton(DataStorageLayout)
    """Data layout configuration."""

    data_repository = providers.Singleton(
        StorageDataRepository,
        storage=_storage,
        data_layout=_data_layout,
    )
    """Data repository service for dataset storage and retrieval."""

    data_processing_pipeline_factory = providers.Singleton(
        NewDataProcessingPipelineFactory,
        feature_extractor=providers.Singleton(FeatureExtractorImpl),
    )
    """Factory for creating data processing pipelines."""

    data_manager = providers.Singleton(
        DataManager,
        data_pipeline_factory=data_processing_pipeline_factory,
        data_repository=data_repository,
        pipeline_executor=providers.Singleton(PipelineExecutor, observers={LoggingObserver()}),
        resource_calculator=resource_calculator,
        resource_settings=settings.provided.resources,
    )
    """Data manager service for handling dataset processing."""


def create_container() -> Container:
    """Create and initialize the DI container.

    Returns:
        Initialized Container instance.
    """
    container = Container()
    return container


container = create_container()
"""Global container instance for application-wide dependency access."""
