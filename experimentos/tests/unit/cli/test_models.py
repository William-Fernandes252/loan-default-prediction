"""Tests for models CLI commands."""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing a CLI test runner."""
    return CliRunner()


class DescribeTrainModelCommand:
    """Tests for the `models train` command."""

    @pytest.fixture
    def models_app(self, mock_model_versioner: MagicMock, mock_settings: MagicMock):
        """Import the app after mocking the container."""
        from experiments.cli.models import app

        return app

    def it_trains_model_with_required_args(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "01912345-6789-7abc-8def-0123456789ab"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        result = runner.invoke(models_app, ["train", "taiwan_credit", "random_forest", "smote"])

        assert result.exit_code == 0
        mock_model_versioner.train_new_version.assert_called_once()

    def it_passes_dataset_to_training_params(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "lending_club", "svm", "random_under_sampling"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.dataset == Dataset.LENDING_CLUB

    def it_passes_model_type_to_training_params(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "taiwan_credit", "xgboost", "baseline"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.model_type == ModelType.XGBOOST

    def it_passes_technique_to_training_params(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "taiwan_credit", "mlp", "smote_tomek"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.technique == Technique.SMOTE_TOMEK

    def it_passes_use_gpu_option(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "taiwan_credit", "svm", "smote", "--use-gpu"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.use_gpu is True

    def it_passes_use_gpu_option_short_form(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "taiwan_credit", "svm", "smote", "-g"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.use_gpu is True

    def it_passes_n_jobs_option(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "taiwan_credit", "svm", "smote", "--n-jobs", "8"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.n_jobs == 8

    def it_passes_n_jobs_option_short_form(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        runner.invoke(models_app, ["train", "taiwan_credit", "svm", "smote", "-j", "4"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.n_jobs == 4

    def it_uses_settings_defaults_when_options_not_provided(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)
        mock_settings.resources.use_gpu = True
        mock_settings.resources.n_jobs = 16

        runner.invoke(models_app, ["train", "taiwan_credit", "svm", "smote"])

        call_args = mock_model_versioner.train_new_version.call_args
        params = call_args[0][0]
        assert params.use_gpu is True
        assert params.n_jobs == 16

    def it_outputs_model_version_id_on_success(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "01912345-6789-7abc-8def-0123456789ab"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        result = runner.invoke(models_app, ["train", "taiwan_credit", "svm", "smote"])

        assert "01912345-6789-7abc-8def-0123456789ab" in result.output

    def it_accepts_all_model_types(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        for model_type in ModelType:
            result = runner.invoke(
                models_app, ["train", "taiwan_credit", model_type.value, "baseline"]
            )
            assert result.exit_code == 0, f"Failed for model type: {model_type.value}"

    def it_accepts_all_techniques(
        self,
        runner: CliRunner,
        mock_model_versioner: MagicMock,
        mock_settings: MagicMock,
        models_app,
    ) -> None:
        mock_version = MagicMock()
        mock_version.id = "test-id"
        mock_model_versioner.train_new_version.return_value = (MagicMock(), mock_version)

        for technique in Technique:
            result = runner.invoke(
                models_app, ["train", "taiwan_credit", "random_forest", technique.value]
            )
            assert result.exit_code == 0, f"Failed for technique: {technique.value}"


class DescribePredictCommand:
    """Tests for the `models predict` command."""

    @pytest.fixture
    def models_app(self, mock_inference_service: MagicMock):
        """Import the app after mocking the container."""
        from experiments.cli.models import app

        return app

    @pytest.fixture
    def output_file(self, tmp_path):
        """Provide a temporary output file path."""
        return str(tmp_path / "predictions.csv")

    def it_runs_inference_with_dataset(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0, 1, 0, 1]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result

        result = runner.invoke(models_app, ["predict", "taiwan_credit", "-o", output_file])

        assert result.exit_code == 0
        mock_inference_service.run_inference_on_test_set.assert_called_once()

    def it_passes_dataset_to_inference_service(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0, 1]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result

        runner.invoke(models_app, ["predict", "lending_club", "-o", output_file])

        call_args = mock_inference_service.run_inference_on_test_set.call_args
        assert call_args[0][0] == Dataset.LENDING_CLUB

    def it_passes_model_id_option(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result
        model_id = "01912345-6789-7abc-8def-0123456789ab"

        runner.invoke(
            models_app, ["predict", "taiwan_credit", "-o", output_file, "--model-id", model_id]
        )

        call_kwargs = mock_inference_service.run_inference_on_test_set.call_args.kwargs
        assert call_kwargs["model_id"] == model_id

    def it_passes_model_id_option_short_form(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [1]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result
        model_id = "01912345-6789-7abc-8def-0123456789ab"

        runner.invoke(models_app, ["predict", "taiwan_credit", "-o", output_file, "-m", model_id])

        call_kwargs = mock_inference_service.run_inference_on_test_set.call_args.kwargs
        assert call_kwargs["model_id"] == model_id

    def it_uses_none_model_id_when_not_specified(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0, 1]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result

        runner.invoke(models_app, ["predict", "taiwan_credit", "-o", output_file])

        call_kwargs = mock_inference_service.run_inference_on_test_set.call_args.kwargs
        assert call_kwargs["model_id"] is None

    def it_outputs_predictions_as_csv(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0, 1, 0, 1, 1]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result

        runner.invoke(models_app, ["predict", "taiwan_credit", "-o", output_file])

        with open(output_file) as f:
            content = f.read()
        assert "prediction" in content  # CSV header
        assert "0" in content
        assert "1" in content

    def it_shows_success_message(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result

        result = runner.invoke(models_app, ["predict", "taiwan_credit", "-o", output_file])

        assert "Predictions saved successfully" in result.output

    def it_accepts_all_datasets(
        self, runner: CliRunner, mock_inference_service: MagicMock, models_app, output_file
    ) -> None:
        mock_result = MagicMock()
        mock_result.predictions = [0]
        mock_inference_service.run_inference_on_test_set.return_value = mock_result

        for dataset in Dataset:
            result = runner.invoke(models_app, ["predict", dataset.value, "-o", output_file])
            assert result.exit_code == 0, f"Failed for dataset: {dataset.value}"
