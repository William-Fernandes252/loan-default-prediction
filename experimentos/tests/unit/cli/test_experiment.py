"""Tests for experiment CLI commands."""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import ModelType


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing a CLI test runner."""
    return CliRunner()


class DescribeExperimentRunCommand:
    """Tests for the `experiment run` command."""

    @pytest.fixture
    def experiment_app(
        self,
        mock_experiment_executor: MagicMock,
        mock_model_predictions_repository: MagicMock,
    ):
        """Import the app after mocking the container."""
        from experiments.cli.experiment import app

        return app

    def it_runs_experiment_with_defaults(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, [])

        assert result.exit_code == 0
        mock_experiment_executor.execute_experiment.assert_called_once()

    def it_passes_only_dataset_option(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["--only-dataset", "taiwan_credit"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.datasets == [Dataset.TAIWAN_CREDIT]

    def it_passes_jobs_option(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["--jobs", "4"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["n_jobs"] == 4

    def it_passes_jobs_option_short_form(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["-j", "2"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["n_jobs"] == 2

    def it_passes_models_jobs_option(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["--models-jobs", "8"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["models_n_jobs"] == 8

    def it_passes_models_jobs_option_short_form(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["-m", "6"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["models_n_jobs"] == 6

    def it_passes_use_gpu_option(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["--use-gpu"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["use_gpu"] is True

    def it_passes_use_gpu_option_short_form(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["-g"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["use_gpu"] is True

    def it_passes_exclude_model_option(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["--exclude-model", "svm"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert ModelType.SVM in params.excluded_models

    def it_passes_exclude_model_option_short_form(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["-x", "random_forest"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert ModelType.RANDOM_FOREST in params.excluded_models

    def it_excludes_multiple_models(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, ["-x", "svm", "-x", "mlp"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert ModelType.SVM in params.excluded_models
        assert ModelType.MLP in params.excluded_models

    def it_passes_execution_id_option(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        mock_experiment_executor.get_completed_count.return_value = 5
        exec_id = "01912345-6789-7abc-8def-0123456789ab"

        result = runner.invoke(experiment_app, ["--execution-id", exec_id])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.execution_id == exec_id

    def it_passes_execution_id_option_short_form(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        mock_experiment_executor.get_completed_count.return_value = 0
        exec_id = "01912345-6789-7abc-8def-0123456789ab"

        result = runner.invoke(experiment_app, ["-e", exec_id])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.execution_id == exec_id

    def it_validates_execution_id_for_continuation(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        mock_experiment_executor.get_completed_count.return_value = 10
        exec_id = "01912345-6789-7abc-8def-0123456789ab"

        result = runner.invoke(experiment_app, ["-e", exec_id])

        assert result.exit_code == 0
        mock_experiment_executor.get_completed_count.assert_called_once_with(exec_id)

    def it_runs_all_datasets_when_none_specified(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(experiment_app, [])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.datasets == list(Dataset)

    def it_combines_multiple_options(
        self, runner: CliRunner, mock_experiment_executor: MagicMock, experiment_app
    ) -> None:
        result = runner.invoke(
            experiment_app,
            [
                "--only-dataset",
                "lending_club",
                "-j",
                "4",
                "-m",
                "8",
                "-g",
                "-x",
                "svm",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        config = call_args[0][1]
        assert params.datasets == [Dataset.LENDING_CLUB]
        assert ModelType.SVM in params.excluded_models
        assert config["n_jobs"] == 4
        assert config["models_n_jobs"] == 8
        assert config["use_gpu"] is True
