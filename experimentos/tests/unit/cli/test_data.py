"""Tests for data CLI commands."""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from experiments.core.data import Dataset


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing a CLI test runner."""
    return CliRunner()


class DescribeProcessDataCommand:
    """Tests for the `data process` command."""

    @pytest.fixture
    def data_app(self, mock_data_manager: MagicMock):
        """Import the app after mocking the container."""
        from experiments.cli.data import app

        return app

    def it_processes_single_dataset(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["taiwan_credit"])

        assert result.exit_code == 0
        mock_data_manager.process_datasets.assert_called_once()
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["datasets"] == [Dataset.TAIWAN_CREDIT]

    def it_processes_all_datasets_when_none_specified(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, [])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["datasets"] == list(Dataset)

    def it_passes_force_flag(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["--force"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["force_overwrite"] is True

    def it_passes_force_flag_short_form(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["-f"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["force_overwrite"] is True

    def it_passes_jobs_option(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["--jobs", "4"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["workers"] == 4

    def it_passes_jobs_option_short_form(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["-j", "2"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["workers"] == 2

    def it_passes_use_gpu_flag(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["--use-gpu"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["use_gpu"] is True

    def it_passes_use_gpu_flag_short_form(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["-g"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["use_gpu"] is True

    def it_exits_with_error_when_processing_fails(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        error = RuntimeError("Processing failed")
        mock_data_manager.process_datasets.return_value = [(Dataset.TAIWAN_CREDIT, error)]

        result = runner.invoke(data_app, ["taiwan_credit"])

        assert result.exit_code == 1

    def it_exits_successfully_when_no_errors(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["taiwan_credit"])

        assert result.exit_code == 0

    def it_accepts_all_valid_datasets(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        for dataset in Dataset:
            result = runner.invoke(data_app, [dataset.value])
            assert result.exit_code == 0, f"Failed for dataset: {dataset.value}"

    def it_combines_multiple_options(
        self, runner: CliRunner, mock_data_manager: MagicMock, data_app
    ) -> None:
        mock_data_manager.process_datasets.return_value = []

        result = runner.invoke(data_app, ["taiwan_credit", "-f", "-j", "8", "-g"])

        assert result.exit_code == 0
        call_kwargs = mock_data_manager.process_datasets.call_args.kwargs
        assert call_kwargs["datasets"] == [Dataset.TAIWAN_CREDIT]
        assert call_kwargs["force_overwrite"] is True
        assert call_kwargs["workers"] == 8
        assert call_kwargs["use_gpu"] is True
