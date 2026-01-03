"""Training pipeline orchestrator.

This module provides the TrainingPipeline class that coordinates
task generation, data loading, training execution, and results persistence.
"""

from dataclasses import dataclass, field
import gc
from typing import Any, Callable

from loguru import logger

from experiments.core.data import Dataset
from experiments.core.modeling.schema import ExperimentConfig
from experiments.core.modeling.types import ModelType
from experiments.core.training.executors import BaseExecutor, ParallelExecutor
from experiments.core.training.generators import (
    ExperimentTaskGenerator,
    TaskGeneratorConfig,
)
from experiments.core.training.persisters import (
    ConsolidationUriProvider,
    ParquetCheckpointPersister,
)
from experiments.core.training.protocols import (
    DataProvider,
    ExperimentRunner,
    ExperimentTask,
)
from experiments.services.storage import StorageService


@dataclass
class TrainingPipelineConfig:
    """Configuration for the training pipeline.

    Attributes:
        cv_folds: Number of cross-validation folds.
        cost_grids: Cost grid configurations.
        num_seeds: Number of random seeds.
        discard_checkpoints: Whether to discard existing checkpoints.
        n_jobs: Number of parallel jobs (None for auto-detection).
        n_jobs_inner: Override for experiment pipeline n_jobs to prevent oversubscription.
    """

    cv_folds: int
    cost_grids: list[Any] = field(default_factory=list)
    num_seeds: int = 30
    discard_checkpoints: bool = False
    n_jobs: int | None = None
    n_jobs_inner: int | None = None


class TrainingPipeline:
    """Orchestrates the training pipeline: Generate → Load → Execute → Persist.

    This class coordinates the training stages using dependency injection
    for maximum flexibility and testability.
    """

    def __init__(
        self,
        task_generator: ExperimentTaskGenerator,
        data_provider: DataProvider,
        executor: BaseExecutor,
        persister: ParquetCheckpointPersister,
        consolidation_provider: ConsolidationUriProvider,
        experiment_runner_factory: Callable[[int | None], ExperimentRunner],
        config: TrainingPipelineConfig,
    ) -> None:
        """Initialize the pipeline.

        Args:
            task_generator: Generator for experiment tasks.
            data_provider: Provider for training data.
            executor: Executor for running training tasks.
            persister: Persister for consolidating results.
            consolidation_provider: Provider for checkpoint and consolidation URIs.
            experiment_runner_factory: Factory function to create experiment runners with specific n_jobs.
            config: Pipeline configuration.
        """
        self._task_generator = task_generator
        self._data_provider = data_provider
        self._executor = executor
        self._persister = persister
        self._consolidation_provider = consolidation_provider
        self._experiment_runner_factory = experiment_runner_factory
        self._config = config

        # Determine n_jobs_inner based on executor type to prevent oversubscription
        self._n_jobs_inner = self._determine_n_jobs_inner()
        self._experiment_runner = self._experiment_runner_factory(self._n_jobs_inner)

    def _determine_n_jobs_inner(self) -> int | None:
        """Determine the n_jobs_inner value to prevent oversubscription.

        Returns:
            1 if using parallel execution to prevent nested parallelism,
            config value if using sequential execution,
            or None to use default.
        """
        if self._config.n_jobs_inner is not None:
            return self._config.n_jobs_inner

        # If using parallel executor, force inner jobs to 1
        if isinstance(self._executor, ParallelExecutor):
            return 1

        # For sequential execution, allow default inner parallelism
        return None

    def _create_experiment_config(self) -> ExperimentConfig:
        """Create experiment configuration from pipeline config."""
        return ExperimentConfig(
            cv_folds=self._config.cv_folds,
            cost_grids=self._config.cost_grids,
            discard_checkpoints=self._config.discard_checkpoints,
        )

    def run(
        self,
        dataset: Dataset,
        excluded_models: set[ModelType] | None = None,
        n_jobs: int | None = None,
    ) -> list[str | None]:
        """Run the training pipeline for a single dataset.

        Args:
            dataset: The dataset to train on.
            excluded_models: Model types to exclude.
            n_jobs: Number of parallel jobs (overrides config).

        Returns:
            List of completed task identifiers.
        """
        # Check data availability
        if not self._data_provider.artifacts_exist(dataset):
            return []

        # Force garbage collection before loading data
        gc.collect()

        # Generate tasks for this dataset
        tasks = self._task_generator.generate([dataset], excluded_models)
        if not tasks:
            logger.warning(f"No tasks generated for {dataset.display_name}")
            return []

        logger.info(f"Dataset {dataset.display_name}: Generated {len(tasks)} tasks")

        # Update executor n_jobs if provided
        if n_jobs is not None and isinstance(self._executor, ParallelExecutor):
            self._executor.n_jobs = n_jobs

        # Create experiment config
        exp_config = self._create_experiment_config()

        # Load data and execute
        try:
            with self._data_provider.feature_context(dataset) as data_paths:
                results = self._executor.execute(
                    tasks=tasks,
                    runner=self._experiment_runner,
                    data_paths=data_paths,
                    config=exp_config,
                    checkpoint_provider=self._consolidation_provider,
                )
        except FileNotFoundError:
            logger.error(f"Data missing for {dataset.display_name}")
            return []

        # Consolidate results
        self._persister.consolidate(dataset)

        return results

    def run_all(
        self,
        datasets: list[Dataset],
        excluded_models: set[ModelType] | None = None,
        compute_jobs_fn: Callable[[float], int] | None = None,
    ) -> dict[str, list[str | None]]:
        """Run the pipeline for multiple datasets.

        Args:
            datasets: List of datasets to process.
            excluded_models: Model types to exclude.
            compute_jobs_fn: Function to compute n_jobs from dataset size in GB.

        Returns:
            Dictionary mapping dataset IDs to their results.
        """
        all_results: dict[str, list[str | None]] = {}

        for dataset in datasets:
            # Compute n_jobs if function provided
            n_jobs = self._config.n_jobs
            if n_jobs is None and compute_jobs_fn is not None:
                size_gb = self._data_provider.get_dataset_size_gb(dataset)
                n_jobs = compute_jobs_fn(size_gb)

            results = self.run(dataset, excluded_models, n_jobs)
            all_results[dataset.id] = results

        return all_results

    def run_single(self, task: ExperimentTask) -> str | None:
        """Run a single experiment task.

        Args:
            task: The experiment task to run.

        Returns:
            Task identifier if successful, None otherwise.
        """
        if not self._data_provider.artifacts_exist(task.dataset):
            return None

        exp_config = self._create_experiment_config()

        try:
            with self._data_provider.feature_context(task.dataset) as data_paths:
                results = self._executor.execute(
                    tasks=[task],
                    runner=self._experiment_runner,
                    data_paths=data_paths,
                    config=exp_config,
                    checkpoint_provider=self._consolidation_provider,
                )
                return results[0] if results else None
        except FileNotFoundError:
            logger.error(f"Data missing for {task.dataset.display_name}")
            return None

    def consolidate(self, dataset: Dataset) -> str | None:
        """Consolidate results for a dataset without running training.

        Args:
            dataset: The dataset to consolidate.

        Returns:
            URI to consolidated results, or None if no results.
        """
        return self._persister.consolidate(dataset)


class TrainingPipelineFactory:
    """Factory for creating pre-configured training pipelines."""

    def __init__(
        self,
        storage: StorageService,
        data_provider: DataProvider,
        consolidation_provider: ConsolidationUriProvider,
        experiment_runner_factory: Callable[[int | None], ExperimentRunner],
    ) -> None:
        """Initialize the factory.

        Args:
            storage: Storage service for file operations.
            data_provider: Provider for training data.
            consolidation_provider: Provider for checkpoint and consolidation URIs.
            experiment_runner_factory: Factory function to create experiment runners with specific n_jobs.
        """
        self._storage = storage
        self._data_provider = data_provider
        self._consolidation_provider = consolidation_provider
        self._experiment_runner_factory = experiment_runner_factory

    def create_parallel_pipeline(
        self,
        config: TrainingPipelineConfig,
        n_jobs: int = -1,
    ) -> TrainingPipeline:
        """Create a pipeline with parallel execution.

        Args:
            config: Pipeline configuration.
            n_jobs: Number of parallel jobs.

        Returns:
            A configured TrainingPipeline with parallel execution.
        """
        task_config = TaskGeneratorConfig(num_seeds=config.num_seeds)

        return TrainingPipeline(
            task_generator=ExperimentTaskGenerator(task_config),
            data_provider=self._data_provider,
            executor=ParallelExecutor(n_jobs=n_jobs),
            persister=ParquetCheckpointPersister(
                storage=self._storage,
                checkpoint_uri_provider=self._consolidation_provider,
                results_uri_provider=self._consolidation_provider,
            ),
            consolidation_provider=self._consolidation_provider,
            experiment_runner_factory=self._experiment_runner_factory,
            config=config,
        )

    def create_sequential_pipeline(
        self,
        config: TrainingPipelineConfig,
    ) -> TrainingPipeline:
        """Create a pipeline with sequential execution.

        Args:
            config: Pipeline configuration.

        Returns:
            A configured TrainingPipeline with sequential execution.
        """
        from experiments.core.training.executors import SequentialExecutor

        task_config = TaskGeneratorConfig(num_seeds=config.num_seeds)

        return TrainingPipeline(
            task_generator=ExperimentTaskGenerator(task_config),
            data_provider=self._data_provider,
            executor=SequentialExecutor(),
            persister=ParquetCheckpointPersister(
                storage=self._storage,
                checkpoint_uri_provider=self._consolidation_provider,
                results_uri_provider=self._consolidation_provider,
            ),
            consolidation_provider=self._consolidation_provider,
            experiment_runner_factory=self._experiment_runner_factory,
            config=config,
        )


__all__ = [
    "TrainingPipelineConfig",
    "TrainingPipeline",
    "TrainingPipelineFactory",
]
