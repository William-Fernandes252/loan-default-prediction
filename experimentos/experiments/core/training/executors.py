"""Training executor implementations for the training pipeline."""

from abc import ABC, abstractmethod
from typing import Any

from joblib import Parallel, delayed
from loguru import logger

from experiments.core.experiment.protocols import (
    DataPaths,
    ExperimentContext,
    ExperimentIdentity,
    TrainingConfig,
)
from experiments.core.training.protocols import (
    CheckpointPathProvider,
    ExperimentRunner,
    ExperimentTask,
)


def create_experiment_context(
    task: ExperimentTask,
    data_paths: tuple[str, str],
    config: Any,
    checkpoint_provider: CheckpointPathProvider,
) -> ExperimentContext:
    """Create experiment context from task and configuration.

    Args:
        task: The experiment task.
        data_paths: Tuple of (X_mmap_path, y_mmap_path).
        config: Experiment configuration.
        checkpoint_provider: Provider for checkpoint URIs.

    Returns:
        Configured ExperimentContext.
    """
    X_mmap_path, y_mmap_path = data_paths

    identity = ExperimentIdentity(
        dataset=task.dataset,
        model_type=task.model_type,
        technique=task.technique,
        seed=task.seed,
    )

    data = DataPaths(
        X_path=X_mmap_path,
        y_path=y_mmap_path,
    )

    training_config = TrainingConfig(
        cv_folds=config.cv_folds,
        cost_grids=config.cost_grids,
    )

    checkpoint_uri = checkpoint_provider.get_checkpoint_uri(
        task.dataset.id,
        task.model_type.id,
        task.technique.id,
        task.seed,
    )

    return ExperimentContext(
        identity=identity,
        data=data,
        config=training_config,
        checkpoint_uri=checkpoint_uri,
        discard_checkpoints=config.discard_checkpoints,
    )


class BaseExecutor(ABC):
    """Base class for training executors."""

    def __init__(self, verbose: int = 5) -> None:
        """Initialize the executor.

        Args:
            verbose: Verbosity level for logging.
        """
        self._verbose = verbose

    @abstractmethod
    def execute(
        self,
        tasks: list[ExperimentTask],
        runner: ExperimentRunner,
        data_paths: tuple[str, str],
        config: Any,
        checkpoint_provider: CheckpointPathProvider,
    ) -> list[str | None]:
        """Execute the training tasks."""
        ...


class SequentialExecutor(BaseExecutor):
    """Executes training tasks sequentially."""

    def execute(
        self,
        tasks: list[ExperimentTask],
        runner: ExperimentRunner,
        data_paths: tuple[str, str],
        config: Any,
        checkpoint_provider: CheckpointPathProvider,
    ) -> list[str | None]:
        """Execute tasks sequentially.

        Args:
            tasks: List of tasks to execute.
            runner: The function to run for each task.
            data_paths: Tuple of (X_mmap_path, y_mmap_path).
            config: Experiment configuration.
            checkpoint_provider: Provider for checkpoint paths.

        Returns:
            List of task identifiers for completed tasks.
        """
        results: list[str | None] = []

        for task in tasks:
            context = create_experiment_context(task, data_paths, config, checkpoint_provider)
            result = runner(context)
            results.append(result.task_id)

        return results


class ParallelExecutor(BaseExecutor):
    """Executes training tasks in parallel using joblib."""

    def __init__(
        self,
        n_jobs: int = -1,
        verbose: int = 5,
        pre_dispatch: str = "2*n_jobs",
    ) -> None:
        """Initialize the parallel executor.

        Args:
            n_jobs: Number of parallel jobs. -1 means use all processors.
            verbose: Verbosity level for joblib.
            pre_dispatch: Pre-dispatch parameter for joblib.
        """
        super().__init__(verbose)
        self._n_jobs = n_jobs
        self._pre_dispatch = pre_dispatch

    @property
    def n_jobs(self) -> int:
        """Get the number of parallel jobs."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        """Set the number of parallel jobs."""
        self._n_jobs = value

    def execute(
        self,
        tasks: list[ExperimentTask],
        runner: ExperimentRunner,
        data_paths: tuple[str, str],
        config: Any,
        checkpoint_provider: CheckpointPathProvider,
    ) -> list[str | None]:
        """Execute tasks in parallel.

        Args:
            tasks: List of tasks to execute.
            runner: The function to run for each task.
            data_paths: Tuple of (X_mmap_path, y_mmap_path).
            config: Experiment configuration.
            checkpoint_provider: Provider for checkpoint paths.

        Returns:
            List of task identifiers for completed tasks.
        """
        logger.info(f"Launching {len(tasks)} tasks with {self._n_jobs} workers...")

        def run_task(task: ExperimentTask) -> str | None:
            context = create_experiment_context(task, data_paths, config, checkpoint_provider)
            result = runner(context)
            return result.task_id

        results = Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            pre_dispatch=self._pre_dispatch,
        )(delayed(run_task)(task) for task in tasks)

        return list(results)


__all__ = [
    "BaseExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
]
