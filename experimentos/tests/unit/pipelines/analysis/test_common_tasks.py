"""Tests for analysis pipeline common tasks."""

from unittest.mock import MagicMock

import pytest

from experiments.core.data import Dataset
from experiments.lib.pipelines.tasks import TaskStatus
from experiments.pipelines.analysis.pipeline import AnalysisPipelineContext, AnalysisPipelineState
from experiments.pipelines.analysis.tasks.common import (
    compute_metrics,
    load_experiment_results,
)
from experiments.services.model_predictions_repository import ExecutionNotFoundError


@pytest.fixture
def mock_predictions_repository() -> MagicMock:
    """Fixture providing a mock predictions repository."""
    return MagicMock()


@pytest.fixture
def mock_results_evaluator() -> MagicMock:
    """Fixture providing a mock results evaluator."""
    return MagicMock()


@pytest.fixture
def mock_artifacts_repository() -> MagicMock:
    """Fixture providing a mock analysis artifacts repository."""
    return MagicMock()


def _make_context(
    predictions_repository: MagicMock,
    results_evaluator: MagicMock,
    artifacts_repository: MagicMock,
    execution_id: str | None = None,
) -> AnalysisPipelineContext:
    """Create a test context with given dependencies."""
    return AnalysisPipelineContext(
        dataset=Dataset.TAIWAN_CREDIT,
        analysis_name="test_analysis",
        predictions_repository=predictions_repository,
        results_evaluator=results_evaluator,
        analysis_artifacts_repository=artifacts_repository,
        use_gpu=False,
        force_overwrite=False,
        translator=None,
        execution_id=execution_id,
    )


class DescribeLoadExperimentResults:
    """Tests for the load_experiment_results task."""

    def it_uses_latest_when_no_execution_id(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        mock_predictions = MagicMock()
        mock_predictions_repository.get_latest_predictions_for_experiment.return_value = (
            mock_predictions
        )

        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
            execution_id=None,
        )

        result = load_experiment_results({}, context)

        mock_predictions_repository.get_latest_predictions_for_experiment.assert_called_once_with(
            Dataset.TAIWAN_CREDIT
        )
        mock_predictions_repository.get_predictions_for_execution.assert_not_called()
        assert result.status == TaskStatus.SUCCESS
        assert result.state["model_predictions"] is mock_predictions

    def it_uses_specific_execution_when_execution_id_provided(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        mock_predictions = MagicMock()
        mock_predictions_repository.get_predictions_for_execution.return_value = mock_predictions

        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
            execution_id="exec-123",
        )

        result = load_experiment_results({}, context)

        mock_predictions_repository.get_predictions_for_execution.assert_called_once_with(
            Dataset.TAIWAN_CREDIT, "exec-123"
        )
        mock_predictions_repository.get_latest_predictions_for_experiment.assert_not_called()
        assert result.status == TaskStatus.SUCCESS
        assert result.state["model_predictions"] is mock_predictions

    def it_returns_failure_when_no_predictions_found(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        mock_predictions_repository.get_latest_predictions_for_experiment.return_value = None

        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
            execution_id=None,
        )

        result = load_experiment_results({}, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "No predictions found" in result.message

    def it_returns_failure_when_specific_execution_has_no_predictions(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        mock_predictions_repository.get_predictions_for_execution.return_value = None

        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
            execution_id="exec-456",
        )

        result = load_experiment_results({}, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "No predictions found" in result.message

    def it_propagates_execution_not_found_error(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        mock_predictions_repository.get_predictions_for_execution.side_effect = (
            ExecutionNotFoundError("nonexistent-exec")
        )

        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
            execution_id="nonexistent-exec",
        )

        with pytest.raises(ExecutionNotFoundError) as exc_info:
            load_experiment_results({}, context)

        assert exc_info.value.execution_id == "nonexistent-exec"


class DescribeComputeMetrics:
    """Tests for the compute_metrics task."""

    def it_computes_metrics_from_predictions(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        mock_predictions = MagicMock()
        mock_metrics = MagicMock()
        mock_results_evaluator.evaluate.return_value = mock_metrics

        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
        )
        state: AnalysisPipelineState = {"model_predictions": mock_predictions}

        result = compute_metrics(state, context)

        mock_results_evaluator.evaluate.assert_called_once_with(mock_predictions)
        assert result.status == TaskStatus.SUCCESS
        assert result.state["metrics"] is mock_metrics

    def it_returns_failure_when_no_predictions_available(
        self,
        mock_predictions_repository: MagicMock,
        mock_results_evaluator: MagicMock,
        mock_artifacts_repository: MagicMock,
    ) -> None:
        context = _make_context(
            mock_predictions_repository,
            mock_results_evaluator,
            mock_artifacts_repository,
        )
        state: AnalysisPipelineState = {"model_predictions": None}

        result = compute_metrics(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message == "No model predictions available to compute metrics."
