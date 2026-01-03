"""Tests for experiments.core.experiment.adapters module."""

from unittest.mock import MagicMock, patch

import pytest

from experiments.core.experiment.adapters import ExperimentRunnerFactory
from experiments.core.experiment.pipeline import (
    ExperimentPipeline,
    ExperimentPipelineConfig,
)
from experiments.core.experiment.protocols import ExperimentResult


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock experiment pipeline."""
    pipeline = MagicMock(spec=ExperimentPipeline)
    pipeline.run.return_value = ExperimentResult(
        task_id="taiwan_credit-random_forest-42",
        metrics={"accuracy_balanced": 0.85},
    )
    return pipeline


class DescribeExperimentRunnerFactory:
    """Tests for ExperimentRunnerFactory class."""

    def it_initializes_with_default_config(self) -> None:
        """Verify factory uses default config when none provided."""
        mock_storage = MagicMock()
        factory = ExperimentRunnerFactory(storage=mock_storage)

        assert factory._pipeline_config == ExperimentPipelineConfig()
        assert factory._storage == mock_storage

    def it_initializes_with_custom_config(self) -> None:
        """Verify factory stores custom config."""
        mock_storage = MagicMock()
        config = ExperimentPipelineConfig(test_size=0.20)
        factory = ExperimentRunnerFactory(storage=mock_storage, pipeline_config=config)

        assert factory._pipeline_config == config


class DescribeExperimentRunnerFactoryCall:
    """Tests for ExperimentRunnerFactory.__call__() method."""

    def it_creates_callable_runner(self) -> None:
        """Verify __call__ returns a callable."""
        mock_storage = MagicMock()
        factory = ExperimentRunnerFactory(storage=mock_storage)

        with patch(
            "experiments.core.experiment.adapters.create_experiment_pipeline"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            runner = factory()

            assert callable(runner)

    def it_uses_injected_storage(self) -> None:
        """Verify factory uses injected storage."""
        mock_storage = MagicMock()
        factory = ExperimentRunnerFactory(storage=mock_storage)

        with patch(
            "experiments.core.experiment.adapters.create_experiment_pipeline"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            factory()

            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert kwargs["storage"] is mock_storage

    def it_returns_pipeline_run_method(self, mock_pipeline: MagicMock) -> None:
        """Verify __call__ returns pipeline.run method."""
        mock_storage = MagicMock()
        factory = ExperimentRunnerFactory(storage=mock_storage)

        with patch(
            "experiments.core.experiment.adapters.create_experiment_pipeline"
        ) as mock_create:
            mock_create.return_value = mock_pipeline

            runner = factory()

            assert runner == mock_pipeline.run

    def it_updates_n_jobs_inner(self) -> None:
        """Verify n_jobs_inner is updated in config."""
        mock_storage = MagicMock()
        factory = ExperimentRunnerFactory(storage=mock_storage)

        with patch(
            "experiments.core.experiment.adapters.create_experiment_pipeline"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            factory(n_jobs_inner=4)

            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert kwargs["config"].n_jobs_inner == 4
