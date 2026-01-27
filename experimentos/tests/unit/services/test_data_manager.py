"""Tests for data_manager service."""

from unittest.mock import MagicMock

import pytest

from experiments.core.data import Dataset
from experiments.services.data_manager import DataManager


class DescribeDataManagerInit:
    def it_stores_all_dependencies(self) -> None:
        pipeline_factory = MagicMock()
        data_repository = MagicMock()
        pipeline_executor = MagicMock()
        resource_calculator = MagicMock()
        resource_settings = MagicMock()

        manager = DataManager(
            data_pipeline_factory=pipeline_factory,
            data_repository=data_repository,
            pipeline_executor=pipeline_executor,
            resource_calculator=resource_calculator,
            resource_settings=resource_settings,
        )

        assert manager._data_pipeline_factory is pipeline_factory
        assert manager._data_repository is data_repository
        assert manager._pipeline_executor is pipeline_executor
        assert manager._resource_calculator is resource_calculator
        assert manager._resource_settings is resource_settings


class DescribeProcessDatasets:
    @pytest.fixture
    def pipeline_factory(self) -> MagicMock:
        factory = MagicMock()
        factory.create.return_value = MagicMock()
        return factory

    @pytest.fixture
    def data_repository(self) -> MagicMock:
        repo = MagicMock()
        repo.get_size_in_bytes.return_value = 1024 * 1024  # 1MB
        return repo

    @pytest.fixture
    def pipeline_executor(self) -> MagicMock:
        executor = MagicMock()
        executor.wait.return_value = []
        return executor

    @pytest.fixture
    def resource_calculator(self) -> MagicMock:
        calc = MagicMock()
        calc.compute_safe_jobs.return_value = 2
        return calc

    @pytest.fixture
    def resource_settings(self) -> MagicMock:
        settings = MagicMock()
        settings.use_gpu = False
        return settings

    @pytest.fixture
    def manager(
        self,
        pipeline_factory: MagicMock,
        data_repository: MagicMock,
        pipeline_executor: MagicMock,
        resource_calculator: MagicMock,
        resource_settings: MagicMock,
    ) -> DataManager:
        return DataManager(
            data_pipeline_factory=pipeline_factory,
            data_repository=data_repository,
            pipeline_executor=pipeline_executor,
            resource_calculator=resource_calculator,
            resource_settings=resource_settings,
        )

    def it_starts_pipeline_executor(
        self,
        manager: DataManager,
        pipeline_executor: MagicMock,
    ) -> None:
        manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        pipeline_executor.start.assert_called_once()

    def it_waits_for_pipeline_results(
        self,
        manager: DataManager,
        pipeline_executor: MagicMock,
    ) -> None:
        manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        pipeline_executor.wait.assert_called_once()

    def it_schedules_pipeline_for_each_dataset(
        self,
        manager: DataManager,
        pipeline_executor: MagicMock,
    ) -> None:
        datasets = [Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB]

        manager.process_datasets(datasets=datasets)

        assert pipeline_executor.schedule.call_count == 2

    def it_returns_empty_list_when_no_errors(
        self,
        manager: DataManager,
        pipeline_executor: MagicMock,
    ) -> None:
        pipeline_executor.wait.return_value = []

        result = manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        assert result == []

    def it_uses_provided_workers_count(
        self,
        manager: DataManager,
        pipeline_executor: MagicMock,
    ) -> None:
        manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT], workers=4)

        pipeline_executor.start.assert_called_with(max_workers=4)

    def it_computes_safe_workers_when_not_provided(
        self,
        manager: DataManager,
        pipeline_executor: MagicMock,
        resource_calculator: MagicMock,
    ) -> None:
        resource_calculator.compute_safe_jobs.return_value = 3

        manager.process_datasets(datasets=[Dataset.TAIWAN_CREDIT])

        # Should use the computed safe jobs value
        pipeline_executor.start.assert_called_with(max_workers=3)


class DescribeGetEffectiveUseGpu:
    @pytest.fixture
    def manager(self) -> DataManager:
        resource_settings = MagicMock()
        resource_settings.use_gpu = True
        return DataManager(
            data_pipeline_factory=MagicMock(),
            data_repository=MagicMock(),
            pipeline_executor=MagicMock(),
            resource_calculator=MagicMock(),
            resource_settings=resource_settings,
        )

    def it_returns_true_when_settings_enable_gpu(
        self,
        manager: DataManager,
    ) -> None:
        result = manager._get_effective_use_gpu(use_gpu=False)

        assert result is True

    def it_returns_true_when_param_enables_gpu(self) -> None:
        resource_settings = MagicMock()
        resource_settings.use_gpu = False
        manager = DataManager(
            data_pipeline_factory=MagicMock(),
            data_repository=MagicMock(),
            pipeline_executor=MagicMock(),
            resource_calculator=MagicMock(),
            resource_settings=resource_settings,
        )

        result = manager._get_effective_use_gpu(use_gpu=True)

        assert result is True


class DescribeGetPipelineName:
    @pytest.fixture
    def manager(self) -> DataManager:
        return DataManager(
            data_pipeline_factory=MagicMock(),
            data_repository=MagicMock(),
            pipeline_executor=MagicMock(),
            resource_calculator=MagicMock(),
            resource_settings=MagicMock(),
        )

    def it_includes_dataset_in_name(self, manager: DataManager) -> None:
        name = manager._get_pipeline_name(Dataset.TAIWAN_CREDIT)

        assert "taiwan_credit" in name.lower() or "Dataset.TAIWAN_CREDIT" in name


class DescribeGetErrorsFromResults:
    @pytest.fixture
    def manager(self) -> DataManager:
        return DataManager(
            data_pipeline_factory=MagicMock(),
            data_repository=MagicMock(),
            pipeline_executor=MagicMock(),
            resource_calculator=MagicMock(),
            resource_settings=MagicMock(),
        )

    def it_returns_empty_list_for_successful_results(
        self,
        manager: DataManager,
    ) -> None:
        context = MagicMock()
        context.dataset = Dataset.TAIWAN_CREDIT
        result = MagicMock()
        result.last_error.return_value = None
        result.context = context

        errors = manager._get_errors_from_results([result])

        assert errors == []

    def it_extracts_errors_from_failed_results(
        self,
        manager: DataManager,
    ) -> None:
        error = RuntimeError("Processing failed")
        context = MagicMock()
        context.dataset = Dataset.TAIWAN_CREDIT
        result = MagicMock()
        result.last_error.return_value = error
        result.context = context

        errors = manager._get_errors_from_results([result])

        assert len(errors) == 1
        assert errors[0][0] == Dataset.TAIWAN_CREDIT
        assert errors[0][1] is error
