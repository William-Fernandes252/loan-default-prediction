"""Tests for data_manager service."""

from unittest.mock import MagicMock

import pytest

from experiments.core.data import Dataset
from experiments.services.data_manager import DataManager

# ============================================================================
# Fixture for DataManager with all mocked dependencies
# ============================================================================


@pytest.fixture
def mock_pipeline_factory() -> MagicMock:
    factory = MagicMock()
    factory.create.return_value = MagicMock()
    return factory


@pytest.fixture
def mock_data_repository() -> MagicMock:
    repo = MagicMock()
    repo.get_size_in_bytes.return_value = 1024 * 1024  # 1MB
    return repo


@pytest.fixture
def mock_pipeline_executor() -> MagicMock:
    executor = MagicMock()
    executor.wait.return_value = []
    return executor


@pytest.fixture
def mock_resource_calculator() -> MagicMock:
    calc = MagicMock()
    calc.compute_safe_jobs.return_value = 2
    return calc


@pytest.fixture
def mock_resource_settings() -> MagicMock:
    settings = MagicMock()
    settings.use_gpu = False
    return settings


@pytest.fixture
def data_manager(
    mock_pipeline_factory: MagicMock,
    mock_data_repository: MagicMock,
    mock_pipeline_executor: MagicMock,
    mock_resource_calculator: MagicMock,
    mock_resource_settings: MagicMock,
) -> DataManager:
    return DataManager(
        data_pipeline_factory=mock_pipeline_factory,
        data_repository=mock_data_repository,
        pipeline_executor=mock_pipeline_executor,
        resource_calculator=mock_resource_calculator,
        resource_settings=mock_resource_settings,
    )


# ============================================================================
# Tests
# ============================================================================


class DescribeDataManagerInit:
    def it_stores_all_dependencies(
        self,
        mock_pipeline_factory: MagicMock,
        mock_data_repository: MagicMock,
        mock_pipeline_executor: MagicMock,
        mock_resource_calculator: MagicMock,
        mock_resource_settings: MagicMock,
    ) -> None:
        manager = DataManager(
            data_pipeline_factory=mock_pipeline_factory,
            data_repository=mock_data_repository,
            pipeline_executor=mock_pipeline_executor,
            resource_calculator=mock_resource_calculator,
            resource_settings=mock_resource_settings,
        )

        assert manager._data_pipeline_factory is mock_pipeline_factory
        assert manager._data_repository is mock_data_repository
        assert manager._pipeline_executor is mock_pipeline_executor
        assert manager._resource_calculator is mock_resource_calculator
        assert manager._resource_settings is mock_resource_settings


class DescribeProcessDatasets:
    def it_starts_pipeline_executor(
        self, data_manager: DataManager, mock_pipeline_executor: MagicMock
    ) -> None:
        data_manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        mock_pipeline_executor.start.assert_called_once()

    def it_waits_for_pipeline_results(
        self, data_manager: DataManager, mock_pipeline_executor: MagicMock
    ) -> None:
        data_manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        mock_pipeline_executor.wait.assert_called_once()

    def it_schedules_pipeline_for_each_dataset(
        self, data_manager: DataManager, mock_pipeline_executor: MagicMock
    ) -> None:
        datasets = [Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB]

        data_manager.process_datasets(datasets=datasets)

        assert mock_pipeline_executor.schedule.call_count == 2

    def it_returns_empty_list_when_no_errors(
        self, data_manager: DataManager, mock_pipeline_executor: MagicMock
    ) -> None:
        mock_pipeline_executor.wait.return_value = []

        result = data_manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        assert result == []

    def it_uses_provided_workers_count(
        self, data_manager: DataManager, mock_pipeline_executor: MagicMock
    ) -> None:
        data_manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT], workers=4)

        mock_pipeline_executor.start.assert_called_with(max_workers=4)

    def it_computes_safe_workers_when_not_provided(
        self,
        data_manager: DataManager,
        mock_pipeline_executor: MagicMock,
        mock_resource_calculator: MagicMock,
    ) -> None:
        mock_resource_calculator.compute_safe_jobs.return_value = 3

        data_manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        mock_pipeline_executor.start.assert_called_with(max_workers=3)


class DescribeGetEffectiveUseGpu:
    def it_returns_true_when_settings_enable_gpu(self) -> None:
        settings = MagicMock()
        settings.use_gpu = True
        manager = DataManager(
            data_pipeline_factory=MagicMock(),
            data_repository=MagicMock(),
            pipeline_executor=MagicMock(),
            resource_calculator=MagicMock(),
            resource_settings=settings,
        )

        result = manager._get_effective_use_gpu(use_gpu=False)

        assert result is True

    def it_returns_true_when_param_enables_gpu(self) -> None:
        settings = MagicMock()
        settings.use_gpu = False
        manager = DataManager(
            data_pipeline_factory=MagicMock(),
            data_repository=MagicMock(),
            pipeline_executor=MagicMock(),
            resource_calculator=MagicMock(),
            resource_settings=settings,
        )

        result = manager._get_effective_use_gpu(use_gpu=True)

        assert result is True


class DescribeGetPipelineName:
    def it_includes_dataset_in_name(self, data_manager: DataManager) -> None:
        name = data_manager._get_pipeline_name(Dataset.TAIWAN_CREDIT)

        assert "taiwan_credit" in name.lower() or "Dataset.TAIWAN_CREDIT" in name


class DescribeGetErrorsFromResults:
    def it_returns_empty_list_for_successful_results(self, data_manager: DataManager) -> None:
        context = MagicMock()
        context.dataset = Dataset.TAIWAN_CREDIT
        result = MagicMock()
        result.last_error.return_value = None
        result.context = context

        errors = data_manager._get_errors_from_results([result])

        assert errors == []

    def it_extracts_errors_from_failed_results(self, data_manager: DataManager) -> None:
        error = RuntimeError("Processing failed")
        context = MagicMock()
        context.dataset = Dataset.TAIWAN_CREDIT
        result = MagicMock()
        result.last_error.return_value = error
        result.context = context

        errors = data_manager._get_errors_from_results([result])

        assert len(errors) == 1
        assert errors[0][0] == Dataset.TAIWAN_CREDIT
        assert errors[0][1] is error
