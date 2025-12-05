"""Training executor implementations for the training pipeline."""

from abc import ABC, abstractmethod
from typing import Any

from joblib import Parallel, delayed
from loguru import logger

from experiments.core.training.protocols import (
    CheckpointPathProvider,
    ExperimentRunner,
    ExperimentTask,
    ModelVersioningProvider,
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
        versioning_provider: ModelVersioningProvider,
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
        versioning_provider: ModelVersioningProvider,
    ) -> list[str | None]:
        """Execute tasks sequentially.

        Args:
            tasks: List of tasks to execute.
            runner: The function to run for each task.
            data_paths: Tuple of (X_mmap_path, y_mmap_path).
            config: Experiment configuration.
            checkpoint_provider: Provider for checkpoint paths.
            versioning_provider: Provider for model versioning services.

        Returns:
            List of task identifiers for completed tasks.
        """
        X_mmap_path, y_mmap_path = data_paths
        results: list[str | None] = []

        for task in tasks:
            checkpoint_path = checkpoint_provider.get_checkpoint_path(
                task.dataset.id,
                task.model_type.id,
                task.technique.id,
                task.seed,
            )
            versioning_service = versioning_provider.get_model_versioning_service(
                task.dataset.id,
                task.model_type,
                task.technique,
            )

            result = runner(
                config,
                task.dataset.id,
                X_mmap_path,
                y_mmap_path,
                task.model_type,
                task.technique,
                task.seed,
                checkpoint_path,
                versioning_service,
            )
            results.append(result)

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
        versioning_provider: ModelVersioningProvider,
    ) -> list[str | None]:
        """Execute tasks in parallel.

        Args:
            tasks: List of tasks to execute.
            runner: The function to run for each task.
            data_paths: Tuple of (X_mmap_path, y_mmap_path).
            config: Experiment configuration.
            checkpoint_provider: Provider for checkpoint paths.
            versioning_provider: Provider for model versioning services.

        Returns:
            List of task identifiers for completed tasks.
        """
        X_mmap_path, y_mmap_path = data_paths

        def _create_task_args(task: ExperimentTask) -> tuple:
            checkpoint_path = checkpoint_provider.get_checkpoint_path(
                task.dataset.id,
                task.model_type.id,
                task.technique.id,
                task.seed,
            )
            versioning_service = versioning_provider.get_model_versioning_service(
                task.dataset.id,
                task.model_type,
                task.technique,
            )

            return (
                config,
                task.dataset.id,
                X_mmap_path,
                y_mmap_path,
                task.model_type,
                task.technique,
                task.seed,
                checkpoint_path,
                versioning_service,
            )

        logger.info(f"Launching {len(tasks)} tasks with {self._n_jobs} workers...")

        results = Parallel(
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            pre_dispatch=self._pre_dispatch,
        )(delayed(runner)(*_create_task_args(task)) for task in tasks)

        return list(results)


__all__ = [
    "BaseExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
]
