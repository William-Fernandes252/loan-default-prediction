"""Tests for experiments.core.training.executors module."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training.executors import (
    BaseExecutor,
    ParallelExecutor,
    SequentialExecutor,
)
from experiments.core.training.protocols import (
    ExperimentContext,
    ExperimentResult,
    ExperimentTask,
)


@pytest.fixture
def sample_tasks() -> list[ExperimentTask]:
    """Create sample experiment tasks."""
    return [
        ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=0,
        ),
        ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=1,
        ),
    ]


@pytest.fixture
def mock_runner() -> MagicMock:
    """Create a mock experiment runner."""
    runner = MagicMock()
    # Return an object with a task_id attribute
    result = MagicMock()
    result.task_id = "task-0"  # Default
    runner.return_value = result

    # Update side effect to return dynamic task_id
    def side_effect(context):
        res = MagicMock()
        res.task_id = f"task-{context.identity.seed}"
        return res

    runner.side_effect = side_effect
    return runner


@pytest.fixture
def mock_checkpoint_provider() -> MagicMock:
    """Create a mock checkpoint provider."""
    provider = MagicMock()
    provider.get_checkpoint_uri.return_value = "/checkpoints/test.parquet"
    return provider


@pytest.fixture
def mock_versioning_provider() -> MagicMock:
    """Create a mock versioning provider."""
    provider = MagicMock()
    provider.get_model_versioning_service.return_value = MagicMock()
    return provider


@pytest.fixture
def sample_data_paths() -> tuple[str, str]:
    """Create sample data paths."""
    return ("/path/X.joblib", "/path/y.joblib")


@pytest.fixture
def sample_config() -> MagicMock:
    """Create a sample config object."""
    return MagicMock()


class DescribeBaseExecutor:
    """Tests for BaseExecutor class."""

    def it_is_abstract(self) -> None:
        """Verify BaseExecutor cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseExecutor()  # type: ignore[abstract]

    def it_stores_verbose_parameter(self) -> None:
        """Verify verbose parameter is stored."""

        class ConcreteExecutor(BaseExecutor):
            def execute(self, *args: Any, **kwargs: Any) -> list[str | None]:
                return []

        executor = ConcreteExecutor(verbose=10)
        assert executor._verbose == 10


class DescribeSequentialExecutor:
    """Tests for SequentialExecutor class."""

    def it_initializes_with_default_verbose(self) -> None:
        """Verify default verbose is 5."""
        executor = SequentialExecutor()

        assert executor._verbose == 5

    def it_accepts_custom_verbose(self) -> None:
        """Verify custom verbose is stored."""
        executor = SequentialExecutor(verbose=10)

        assert executor._verbose == 10


class DescribeSequentialExecutorExecute:
    """Tests for SequentialExecutor.execute() method."""

    def it_executes_all_tasks(
        self,
        sample_tasks: list[ExperimentTask],
        mock_runner: MagicMock,
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify all tasks are executed."""
        executor = SequentialExecutor()

        results = executor.execute(
            tasks=sample_tasks,
            runner=mock_runner,
            data_paths=sample_data_paths,
            config=sample_config,
            checkpoint_provider=mock_checkpoint_provider,
        )

        assert len(results) == 2
        assert mock_runner.call_count == 2

    def it_returns_runner_results(
        self,
        sample_tasks: list[ExperimentTask],
        mock_runner: MagicMock,
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify runner results are returned."""
        executor = SequentialExecutor()

        results = executor.execute(
            tasks=sample_tasks,
            runner=mock_runner,
            data_paths=sample_data_paths,
            config=sample_config,
            checkpoint_provider=mock_checkpoint_provider,
        )

        assert results == ["task-0", "task-1"]

    def it_passes_correct_arguments_to_runner(
        self,
        sample_tasks: list[ExperimentTask],
        mock_runner: MagicMock,
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify correct arguments are passed to runner."""
        executor = SequentialExecutor()

        executor.execute(
            tasks=sample_tasks,
            runner=mock_runner,
            data_paths=sample_data_paths,
            config=sample_config,
            checkpoint_provider=mock_checkpoint_provider,
        )

        # Check first call arguments
        first_call = mock_runner.call_args_list[0]
        context = first_call[0][0]

        assert context.config.cv_folds == sample_config.cv_folds
        assert context.identity.dataset == sample_tasks[0].dataset
        assert context.data.X_path == "/path/X.joblib"
        assert context.data.y_path == "/path/y.joblib"
        assert context.identity.model_type == sample_tasks[0].model_type
        assert context.identity.technique == sample_tasks[0].technique
        assert context.identity.seed == sample_tasks[0].seed

    def it_calls_checkpoint_provider_for_each_task(
        self,
        sample_tasks: list[ExperimentTask],
        mock_runner: MagicMock,
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify checkpoint provider is called for each task."""
        executor = SequentialExecutor()

        executor.execute(
            tasks=sample_tasks,
            runner=mock_runner,
            data_paths=sample_data_paths,
            config=sample_config,
            checkpoint_provider=mock_checkpoint_provider,
        )

        assert mock_checkpoint_provider.get_checkpoint_uri.call_count == 2


class DescribeParallelExecutor:
    """Tests for ParallelExecutor class."""

    def it_initializes_with_default_parameters(self) -> None:
        """Verify default parameters."""
        executor = ParallelExecutor()

        assert executor._n_jobs == -1
        assert executor._verbose == 5
        assert executor._pre_dispatch == "2*n_jobs"

    def it_accepts_custom_parameters(self) -> None:
        """Verify custom parameters are stored."""
        executor = ParallelExecutor(
            n_jobs=4,
            verbose=10,
            pre_dispatch="4*n_jobs",
        )

        assert executor._n_jobs == 4
        assert executor._verbose == 10
        assert executor._pre_dispatch == "4*n_jobs"

    def it_has_n_jobs_property(self) -> None:
        """Verify n_jobs property returns value."""
        executor = ParallelExecutor(n_jobs=4)

        assert executor.n_jobs == 4

    def it_allows_setting_n_jobs(self) -> None:
        """Verify n_jobs can be updated."""
        executor = ParallelExecutor(n_jobs=4)
        executor.n_jobs = 8

        assert executor.n_jobs == 8


class DescribeParallelExecutorExecute:
    """Tests for ParallelExecutor.execute() method."""

    def it_executes_all_tasks(
        self,
        sample_tasks: list[ExperimentTask],
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify all tasks are executed."""
        executor = ParallelExecutor(n_jobs=1)

        # Create a simple runner that returns task IDs
        def simple_runner(context: ExperimentContext) -> ExperimentResult:
            result = MagicMock(spec=ExperimentResult)
            result.task_id = f"task-{context.identity.seed}"
            return result

        results = executor.execute(
            tasks=sample_tasks,
            runner=simple_runner,
            data_paths=sample_data_paths,
            config=sample_config,
            checkpoint_provider=mock_checkpoint_provider,
        )

        assert len(results) == 2
        assert "task-0" in results
        assert "task-1" in results

    def it_uses_joblib_parallel(
        self,
        sample_tasks: list[ExperimentTask],
        mock_runner: MagicMock,
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify joblib.Parallel is used."""
        executor = ParallelExecutor(n_jobs=2)

        with patch("experiments.core.training.executors.Parallel") as mock_parallel_class:
            mock_parallel = MagicMock()
            mock_parallel.return_value = ["task-0", "task-1"]
            mock_parallel_class.return_value = mock_parallel

            executor.execute(
                tasks=sample_tasks,
                runner=mock_runner,
                data_paths=sample_data_paths,
                config=sample_config,
                checkpoint_provider=mock_checkpoint_provider,
            )

            mock_parallel_class.assert_called_once_with(
                n_jobs=2,
                verbose=5,
                pre_dispatch="2*n_jobs",
            )

    def it_returns_list_of_results(
        self,
        sample_tasks: list[ExperimentTask],
        sample_data_paths: tuple[str, str],
        sample_config: MagicMock,
        mock_checkpoint_provider: MagicMock,
    ) -> None:
        """Verify results are returned as list."""
        executor = ParallelExecutor(n_jobs=1)

        def simple_runner(context: ExperimentContext) -> ExperimentResult:
            result = MagicMock(spec=ExperimentResult)
            result.task_id = f"task-{context.identity.seed}"
            return result

        results = executor.execute(
            tasks=sample_tasks,
            runner=simple_runner,
            data_paths=sample_data_paths,
            config=sample_config,
            checkpoint_provider=mock_checkpoint_provider,
        )

        assert isinstance(results, list)
