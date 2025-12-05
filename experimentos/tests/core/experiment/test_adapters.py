"""Tests for experiments.core.experiment.adapters module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.core.data import Dataset
from experiments.core.experiment.adapters import (
    ExperimentRunnerFactory,
    create_experiment_runner,
)
from experiments.core.experiment.pipeline import (
    ExperimentPipeline,
    ExperimentPipelineConfig,
)
from experiments.core.experiment.protocols import ExperimentResult
from experiments.core.modeling.schema import ExperimentConfig
from experiments.core.modeling.types import ModelType, Technique


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock experiment pipeline."""
    pipeline = MagicMock(spec=ExperimentPipeline)
    pipeline.run.return_value = ExperimentResult(
        task_id="taiwan_credit-random_forest-42",
        metrics={"accuracy_balanced": 0.85},
    )
    return pipeline


@pytest.fixture
def sample_experiment_config() -> ExperimentConfig:
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        cv_folds=5,
        cost_grids=[],
        discard_checkpoints=False,
    )


class DescribeCreateExperimentRunner:
    """Tests for create_experiment_runner function."""

    def it_returns_callable(self, mock_pipeline: MagicMock) -> None:
        """Verify function returns a callable runner."""
        runner = create_experiment_runner(mock_pipeline)

        assert callable(runner)

    def it_runner_calls_pipeline_run(
        self,
        mock_pipeline: MagicMock,
        sample_experiment_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        """Verify runner calls pipeline.run with correct context."""
        runner = create_experiment_runner(mock_pipeline)

        result = runner(
            cfg=sample_experiment_config,
            dataset_val="taiwan_credit",
            X_mmap_path="/path/X.joblib",
            y_mmap_path="/path/y.joblib",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            checkpoint_path=tmp_path / "checkpoint.parquet",
        )

        mock_pipeline.run.assert_called_once()
        assert result == "taiwan_credit-random_forest-42"

    def it_runner_returns_task_id(
        self,
        mock_pipeline: MagicMock,
        sample_experiment_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        """Verify runner returns task_id from ExperimentResult."""
        runner = create_experiment_runner(mock_pipeline)

        result = runner(
            cfg=sample_experiment_config,
            dataset_val="taiwan_credit",
            X_mmap_path="/path/X.joblib",
            y_mmap_path="/path/y.joblib",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            checkpoint_path=tmp_path / "checkpoint.parquet",
        )

        assert result == "taiwan_credit-random_forest-42"

    def it_runner_returns_none_when_experiment_skipped(
        self,
        sample_experiment_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        """Verify runner returns None when experiment is skipped."""
        mock_pipeline = MagicMock(spec=ExperimentPipeline)
        mock_pipeline.run.return_value = ExperimentResult(
            task_id=None,
            metrics={},
        )

        runner = create_experiment_runner(mock_pipeline)

        result = runner(
            cfg=sample_experiment_config,
            dataset_val="taiwan_credit",
            X_mmap_path="/path/X.joblib",
            y_mmap_path="/path/y.joblib",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            checkpoint_path=tmp_path / "checkpoint.parquet",
        )

        assert result is None

    def it_runner_creates_correct_experiment_context(
        self,
        mock_pipeline: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify runner creates ExperimentContext with correct values."""
        config = ExperimentConfig(
            cv_folds=10,
            cost_grids=[{"cost": [1, 10]}],
            discard_checkpoints=True,
        )

        runner = create_experiment_runner(mock_pipeline)
        checkpoint_path = tmp_path / "checkpoint.parquet"

        runner(
            cfg=config,
            dataset_val="taiwan_credit",
            X_mmap_path="/path/X.joblib",
            y_mmap_path="/path/y.joblib",
            model_type=ModelType.SVM,
            technique=Technique.SMOTE,
            seed=123,
            checkpoint_path=checkpoint_path,
        )

        # Inspect the context passed to pipeline.run
        call_args = mock_pipeline.run.call_args
        context = call_args[0][0]

        assert context.dataset == Dataset.TAIWAN_CREDIT
        assert context.model_type == ModelType.SVM
        assert context.technique == Technique.SMOTE
        assert context.seed == 123
        assert context.cv_folds == 10
        assert context.cost_grids == [{"cost": [1, 10]}]
        assert context.checkpoint_path == checkpoint_path
        assert context.discard_checkpoints is True

    def it_runner_resolves_dataset_from_string_id(
        self,
        mock_pipeline: MagicMock,
        sample_experiment_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        """Verify runner resolves dataset from string ID."""
        runner = create_experiment_runner(mock_pipeline)

        runner(
            cfg=sample_experiment_config,
            dataset_val="lending_club",
            X_mmap_path="/path/X.joblib",
            y_mmap_path="/path/y.joblib",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            checkpoint_path=tmp_path / "checkpoint.parquet",
        )

        call_args = mock_pipeline.run.call_args
        context = call_args[0][0]
        assert context.dataset == Dataset.LENDING_CLUB

    def it_runner_passes_mmap_paths_to_pipeline(
        self,
        mock_pipeline: MagicMock,
        sample_experiment_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        """Verify runner passes mmap paths to pipeline.run."""
        runner = create_experiment_runner(mock_pipeline)

        runner(
            cfg=sample_experiment_config,
            dataset_val="taiwan_credit",
            X_mmap_path="/custom/X.joblib",
            y_mmap_path="/custom/y.joblib",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            checkpoint_path=tmp_path / "checkpoint.parquet",
        )

        call_args = mock_pipeline.run.call_args
        assert call_args[0][1] == "/custom/X.joblib"
        assert call_args[0][2] == "/custom/y.joblib"


class DescribeExperimentRunnerFactory:
    """Tests for ExperimentRunnerFactory class."""

    def it_initializes_with_default_config(self) -> None:
        """Verify factory uses default config when none provided."""
        factory = ExperimentRunnerFactory()

        assert factory._pipeline_config == ExperimentPipelineConfig()

    def it_initializes_with_custom_config(self) -> None:
        """Verify factory stores custom config."""
        config = ExperimentPipelineConfig(test_size=0.20)
        factory = ExperimentRunnerFactory(pipeline_config=config)

        assert factory._pipeline_config == config


class DescribeExperimentRunnerFactoryCreateRunner:
    """Tests for ExperimentRunnerFactory.create_runner() method."""

    def it_creates_callable_runner(self) -> None:
        """Verify create_runner returns a callable."""
        factory = ExperimentRunnerFactory()

        with patch(
            "experiments.core.experiment.adapters.ExperimentPipelineFactory"
        ) as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_default_pipeline.return_value = MagicMock()
            mock_factory_class.return_value = mock_factory

            runner = factory.create_runner()

            assert callable(runner)

    def it_passes_versioning_service_to_pipeline_factory(self) -> None:
        """Verify versioning service is passed to pipeline factory."""
        factory = ExperimentRunnerFactory()
        mock_service = MagicMock()

        with patch(
            "experiments.core.experiment.adapters.ExperimentPipelineFactory"
        ) as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create_default_pipeline.return_value = MagicMock()
            mock_factory_class.return_value = mock_factory

            factory.create_runner(model_versioning_service=mock_service)

            mock_factory_class.assert_called_once_with(model_versioning_service=mock_service)


class DescribeExperimentRunnerFactoryCreateRunnerWithPipeline:
    """Tests for ExperimentRunnerFactory.create_runner_with_pipeline() method."""

    def it_creates_runner_from_existing_pipeline(self, mock_pipeline: MagicMock) -> None:
        """Verify runner is created from existing pipeline."""
        factory = ExperimentRunnerFactory()

        runner = factory.create_runner_with_pipeline(mock_pipeline)

        assert callable(runner)

    def it_runner_uses_provided_pipeline(
        self,
        mock_pipeline: MagicMock,
        sample_experiment_config: ExperimentConfig,
        tmp_path: Path,
    ) -> None:
        """Verify runner uses the provided pipeline."""
        factory = ExperimentRunnerFactory()
        runner = factory.create_runner_with_pipeline(mock_pipeline)

        runner(
            cfg=sample_experiment_config,
            dataset_val="taiwan_credit",
            X_mmap_path="/path/X.joblib",
            y_mmap_path="/path/y.joblib",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            checkpoint_path=tmp_path / "checkpoint.parquet",
        )

        mock_pipeline.run.assert_called_once()
