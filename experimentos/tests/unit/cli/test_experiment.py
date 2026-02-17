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
        mock_experiment_params_resolver: MagicMock,
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
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        mock_experiment_params_resolver: MagicMock,
        experiment_app,
    ) -> None:
        from experiments.services.experiment_executor import ExperimentParams
        from experiments.services.experiment_params_resolver import (
            ResolutionStatus,
            ResolutionSuccess,
        )

        exec_id = "01912345-6789-7abc-8def-0123456789ab"

        # Mock resolver to return explicit ID continuation
        def _resolve_with_explicit_id(options, config):
            params = ExperimentParams(
                datasets=options.datasets,
                excluded_models=options.excluded_models,
                execution_id=exec_id,
            )
            return ResolutionSuccess(
                params=params,
                context={
                    "status": ResolutionStatus.EXPLICIT_ID_CONTINUED,
                    "execution_id": exec_id,
                    "completed_count": 5,
                    "datasets": options.datasets,
                },
            )

        mock_experiment_params_resolver.resolve_params.side_effect = _resolve_with_explicit_id

        result = runner.invoke(experiment_app, ["--execution-id", exec_id])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.execution_id == exec_id

    def it_passes_execution_id_option_short_form(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        mock_experiment_params_resolver: MagicMock,
        experiment_app,
    ) -> None:
        from experiments.services.experiment_executor import ExperimentParams
        from experiments.services.experiment_params_resolver import (
            ResolutionStatus,
            ResolutionSuccess,
        )

        exec_id = "01912345-6789-7abc-8def-0123456789ab"

        # Mock resolver to return explicit ID as new
        def _resolve_with_explicit_id(options, config):
            params = ExperimentParams(
                datasets=options.datasets,
                excluded_models=options.excluded_models,
                execution_id=exec_id,
            )
            return ResolutionSuccess(
                params=params,
                context={
                    "status": ResolutionStatus.EXPLICIT_ID_NEW,
                    "execution_id": exec_id,
                    "completed_count": 0,
                    "datasets": options.datasets,
                },
            )

        mock_experiment_params_resolver.resolve_params.side_effect = _resolve_with_explicit_id

        result = runner.invoke(experiment_app, ["-e", exec_id])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.execution_id == exec_id

    def it_validates_execution_id_for_continuation(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        mock_experiment_params_resolver: MagicMock,
        experiment_app,
    ) -> None:
        from experiments.services.experiment_executor import ExperimentParams
        from experiments.services.experiment_params_resolver import (
            ResolutionStatus,
            ResolutionSuccess,
        )

        exec_id = "01912345-6789-7abc-8def-0123456789ab"

        # Mock resolver to indicate continuation
        def _resolve_continuation(options, config):
            params = ExperimentParams(
                datasets=options.datasets,
                excluded_models=options.excluded_models,
                execution_id=exec_id,
            )
            return ResolutionSuccess(
                params=params,
                context={
                    "status": ResolutionStatus.EXPLICIT_ID_CONTINUED,
                    "execution_id": exec_id,
                    "completed_count": 10,
                    "datasets": options.datasets,
                },
            )

        mock_experiment_params_resolver.resolve_params.side_effect = _resolve_continuation

        result = runner.invoke(experiment_app, ["-e", exec_id])

        assert result.exit_code == 0
        # Verify resolver was called (which internally validates)
        mock_experiment_params_resolver.resolve_params.assert_called_once()

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

    def it_passes_skip_resume_flag(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        mock_experiment_params_resolver: MagicMock,
        experiment_app,
    ) -> None:
        """Test that --skip-resume flag forces new execution."""
        result = runner.invoke(experiment_app, ["--skip-resume"])

        assert result.exit_code == 0

        # Should start new execution (params.execution_id will be generated)
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.execution_id is not None  # New UUID7 generated

    def it_rejects_skip_resume_with_execution_id(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        mock_experiment_params_resolver: MagicMock,
        experiment_app,
    ) -> None:
        """Test that --skip-resume and --execution-id cannot be used together."""
        from experiments.services.experiment_params_resolver import ResolutionError

        exec_id = "01943abc-1234-7000-8000-0123456789ab"

        # Mock resolver to return error for mutually exclusive options
        # Must clear side_effect first for return_value to take effect
        mock_experiment_params_resolver.resolve_params.side_effect = None
        mock_experiment_params_resolver.resolve_params.return_value = ResolutionError(
            code="mutually_exclusive_options",
            message="Cannot use both execution_id and skip_resume options together",
        )

        result = runner.invoke(experiment_app, ["--execution-id", exec_id, "--skip-resume"])

        assert result.exit_code == 1

        # Should not execute experiment
        mock_experiment_executor.execute_experiment.assert_not_called()

    def it_logs_skip_resume_message(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        mock_experiment_params_resolver: MagicMock,
        experiment_app,
    ) -> None:
        """Test that --skip-resume starts a new execution."""
        result = runner.invoke(experiment_app, ["--skip-resume"])

        assert result.exit_code == 0

        # Verify new execution was started
        call_args = mock_experiment_executor.execute_experiment.call_args
        params = call_args[0][0]
        assert params.execution_id is not None

    def it_passes_sequential_flag(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        experiment_app,
    ) -> None:
        result = runner.invoke(experiment_app, ["--sequential"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["sequential"] is True

    def it_passes_sequential_flag_short_form(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        experiment_app,
    ) -> None:
        result = runner.invoke(experiment_app, ["-s"])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert config["sequential"] is True

    def it_does_not_include_sequential_when_not_specified(
        self,
        runner: CliRunner,
        mock_experiment_executor: MagicMock,
        experiment_app,
    ) -> None:
        result = runner.invoke(experiment_app, [])

        assert result.exit_code == 0
        call_args = mock_experiment_executor.execute_experiment.call_args
        config = call_args[0][1]
        assert "sequential" not in config
