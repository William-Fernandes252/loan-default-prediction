"""Shared fixtures for CLI unit tests."""

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_container() -> Generator[MagicMock, None, None]:
    """Fixture providing a mocked DI container.

    Patches the container in all CLI modules to ensure mock is used.
    """
    mock = MagicMock()

    with (
        patch("experiments.cli.data.container", mock),
        patch("experiments.cli.experiment.container", mock),
        patch("experiments.cli.models.container", mock),
        patch("experiments.cli.analysis.container", mock),
    ):
        yield mock


@pytest.fixture
def mock_data_manager(mock_container: MagicMock) -> MagicMock:
    """Fixture providing a mocked DataManager from the container."""
    manager = MagicMock()
    mock_container.data_manager.return_value = manager
    return manager


@pytest.fixture
def mock_experiment_executor(mock_container: MagicMock) -> MagicMock:
    """Fixture providing a mocked ExperimentExecutor from the container."""
    executor = MagicMock()
    mock_container.experiment_executor.return_value = executor
    return executor


@pytest.fixture
def mock_model_predictions_repository(mock_container: MagicMock) -> MagicMock:
    """Fixture providing a mocked ModelPredictionsRepository from the container."""
    repository = MagicMock()
    # Default to returning None for get_latest_execution_id to simulate no prior executions
    repository.get_latest_execution_id.return_value = None
    mock_container.model_predictions_repository.return_value = repository
    return repository


@pytest.fixture
def mock_model_versioner(mock_container: MagicMock) -> MagicMock:
    """Fixture providing a mocked ModelVersioner from the container."""
    versioner = MagicMock()
    mock_container.model_versioner.return_value = versioner
    return versioner


@pytest.fixture
def mock_inference_service(mock_container: MagicMock) -> MagicMock:
    """Fixture providing a mocked InferenceService from the container."""
    service = MagicMock()
    mock_container.inference_service.return_value = service
    return service


@pytest.fixture
def mock_settings(mock_container: MagicMock) -> MagicMock:
    """Fixture providing mocked settings from the container."""
    settings = MagicMock()
    settings.resources.use_gpu = False
    settings.resources.n_jobs = 4
    mock_container.settings.return_value = settings
    return settings
