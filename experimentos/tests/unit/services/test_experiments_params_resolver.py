"""Tests for experiment_params_resolver service."""

from unittest.mock import ANY, MagicMock

import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType
from experiments.core.predictions.repository import ModelPredictionsRepository
from experiments.services.experiment_executor import ExperimentConfig, ExperimentExecutor
from experiments.services.experiment_params_resolver import (
    ExperimentParamsResolver,
    ResolutionError,
    ResolutionStatus,
    ResolutionSuccess,
    ResolverOptions,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_predictions_repository() -> MagicMock:
    """Mock ModelPredictionsRepository for testing."""
    return MagicMock(spec=ModelPredictionsRepository)


@pytest.fixture
def mock_experiment_executor() -> MagicMock:
    """Mock ExperimentExecutor for testing."""
    return MagicMock(spec=ExperimentExecutor)


@pytest.fixture
def resolver(
    mock_predictions_repository: MagicMock,
    mock_experiment_executor: MagicMock,
) -> ExperimentParamsResolver:
    """ExperimentParamsResolver instance with mocked dependencies."""
    return ExperimentParamsResolver(
        mock_predictions_repository,
        mock_experiment_executor,
    )


# ============================================================================
# Helpers
# ============================================================================


def _make_options(
    *,
    execution_id: str | None = None,
    skip_resume: bool = False,
    datasets: list[Dataset] | None = None,
    excluded_models: list[ModelType] | None = None,
) -> ResolverOptions:
    """Create ResolverOptions with sensible defaults."""
    return ResolverOptions(
        datasets=datasets or [Dataset.TAIWAN_CREDIT],
        excluded_models=excluded_models or [],
        execution_id=execution_id,
        skip_resume=skip_resume,
    )


class DescribeResolveParams:
    def it_returns_error_when_execution_id_and_skip_resume_are_both_set(
        self,
        resolver: ExperimentParamsResolver,
        mock_predictions_repository: MagicMock,
        mock_experiment_executor: MagicMock,
    ) -> None:
        options = _make_options(execution_id="exec-1", skip_resume=True)

        result = resolver.resolve_params(options, config={})

        assert isinstance(result, ResolutionError)
        assert result.code == "mutually_exclusive_options"
        mock_predictions_repository.get_latest_execution_id.assert_not_called()
        mock_experiment_executor.get_completed_count.assert_not_called()
        mock_experiment_executor.is_execution_complete.assert_not_called()


class DescribeResolveParamsWithExplicitId:
    def it_marks_execution_as_new_when_no_prior_work(
        self,
        resolver: ExperimentParamsResolver,
        mock_experiment_executor: MagicMock,
        mock_predictions_repository: MagicMock,
    ) -> None:
        mock_experiment_executor.get_completed_count.return_value = 0
        options = _make_options(execution_id="exec-42")

        result = resolver.resolve_params(options, config={})

        assert isinstance(result, ResolutionSuccess)
        assert result.context["status"] == ResolutionStatus.EXPLICIT_ID_NEW
        assert result.context["execution_id"] == "exec-42"
        assert result.params.execution_id == "exec-42"
        mock_experiment_executor.get_completed_count.assert_called_once_with("exec-42")
        mock_predictions_repository.get_latest_execution_id.assert_not_called()

    def it_marks_execution_as_continued_when_prior_work_exists(
        self,
        resolver: ExperimentParamsResolver,
        mock_experiment_executor: MagicMock,
        mock_predictions_repository: MagicMock,
    ) -> None:
        mock_experiment_executor.get_completed_count.return_value = 3
        options = _make_options(execution_id="exec-99")

        result = resolver.resolve_params(options, config={})

        assert isinstance(result, ResolutionSuccess)
        assert result.context["completed_count"] == 3
        assert result.params.execution_id == "exec-99"
        mock_experiment_executor.get_completed_count.assert_called_once_with("exec-99")
        mock_predictions_repository.get_latest_execution_id.assert_not_called()


class DescribeResolveParamsWithSkipResume:
    def it_forces_new_execution_without_querying_history(
        self,
        resolver: ExperimentParamsResolver,
        mock_predictions_repository: MagicMock,
        mock_experiment_executor: MagicMock,
    ) -> None:
        options = _make_options(skip_resume=True)

        result = resolver.resolve_params(options, config={})

        assert isinstance(result, ResolutionSuccess)
        assert result.context["status"] == ResolutionStatus.SKIP_RESUME
        assert result.context["execution_id"] == result.params.execution_id
        assert result.params.execution_id
        mock_predictions_repository.get_latest_execution_id.assert_not_called()
        mock_experiment_executor.get_completed_count.assert_not_called()
        mock_experiment_executor.is_execution_complete.assert_not_called()


class DescribeResolveParamsWithAutoResume:
    def it_starts_new_execution_when_no_prior_execution_found(
        self,
        resolver: ExperimentParamsResolver,
        mock_predictions_repository: MagicMock,
        mock_experiment_executor: MagicMock,
    ) -> None:
        mock_predictions_repository.get_latest_execution_id.return_value = None
        options = _make_options()

        result = resolver.resolve_params(options, config={})

        assert isinstance(result, ResolutionSuccess)
        assert result.context["status"] == ResolutionStatus.NEW_EXECUTION
        assert result.params.execution_id == result.context["execution_id"]
        mock_predictions_repository.get_latest_execution_id.assert_called_once_with(
            options.datasets
        )
        mock_experiment_executor.is_execution_complete.assert_not_called()
        mock_experiment_executor.get_completed_count.assert_not_called()

    def it_returns_already_complete_when_latest_execution_is_complete(
        self,
        resolver: ExperimentParamsResolver,
        mock_predictions_repository: MagicMock,
        mock_experiment_executor: MagicMock,
    ) -> None:
        mock_predictions_repository.get_latest_execution_id.return_value = "exec-7"
        mock_experiment_executor.is_execution_complete.return_value = True
        options = _make_options()
        config: ExperimentConfig = {"num_seeds": 2}

        result = resolver.resolve_params(options, config=config)

        assert isinstance(result, ResolutionSuccess)
        assert result.context["status"] == ResolutionStatus.ALREADY_COMPLETE
        assert result.should_exit_early is True
        assert result.params.execution_id == "exec-7"
        mock_experiment_executor.is_execution_complete.assert_called_once_with(
            "exec-7",
            ANY,
            config,
        )
        mock_experiment_executor.get_completed_count.assert_not_called()

    def it_resumes_incomplete_execution_when_latest_is_not_complete(
        self,
        resolver: ExperimentParamsResolver,
        mock_predictions_repository: MagicMock,
        mock_experiment_executor: MagicMock,
    ) -> None:
        mock_predictions_repository.get_latest_execution_id.return_value = "exec-8"
        mock_experiment_executor.is_execution_complete.return_value = False
        mock_experiment_executor.get_completed_count.return_value = 5
        options = _make_options(datasets=[Dataset.LENDING_CLUB])

        result = resolver.resolve_params(options, config={})

        assert isinstance(result, ResolutionSuccess)
        assert result.context["status"] == ResolutionStatus.RESUMED_INCOMPLETE
        assert result.context["completed_count"] == 5
        assert result.params.execution_id == "exec-8"
        mock_predictions_repository.get_latest_execution_id.assert_called_once_with(
            options.datasets
        )
        mock_experiment_executor.is_execution_complete.assert_called_once_with(
            "exec-8",
            ANY,
            {},
        )
        mock_experiment_executor.get_completed_count.assert_called_once_with("exec-8")
