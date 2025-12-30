"""Tests for experiments.core.training.pipeline module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training.executors import ParallelExecutor, SequentialExecutor
from experiments.core.training.generators import ExperimentTaskGenerator
from experiments.core.training.persisters import ParquetCheckpointPersister
from experiments.core.training.pipeline import (
    TrainingPipeline,
    TrainingPipelineConfig,
    TrainingPipelineFactory,
)
from experiments.core.training.protocols import ExperimentTask
from experiments.services.storage import StorageService
from experiments.services.storage.local import LocalStorageService


@pytest.fixture
def storage() -> StorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


class DescribeTrainingPipelineConfig:
    """Tests for TrainingPipelineConfig class."""

    def it_requires_cv_folds(self) -> None:
        """Verify cv_folds is required."""
        config = TrainingPipelineConfig(cv_folds=5)

        assert config.cv_folds == 5

    def it_has_default_num_seeds(self) -> None:
        """Verify default num_seeds is 30."""
        config = TrainingPipelineConfig(cv_folds=5)

        assert config.num_seeds == 30

    def it_accepts_custom_num_seeds(self) -> None:
        """Verify custom num_seeds is stored."""
        config = TrainingPipelineConfig(cv_folds=5, num_seeds=50)

        assert config.num_seeds == 50

    def it_has_empty_cost_grids_by_default(self) -> None:
        """Verify cost_grids defaults to empty list."""
        config = TrainingPipelineConfig(cv_folds=5)

        assert config.cost_grids == []

    def it_accepts_custom_cost_grids(self) -> None:
        """Verify custom cost_grids is stored."""
        grids: list[Any] = [{"C": [1, 10]}]
        config = TrainingPipelineConfig(cv_folds=5, cost_grids=grids)

        assert config.cost_grids == grids

    def it_has_false_discard_checkpoints_by_default(self) -> None:
        """Verify discard_checkpoints defaults to False."""
        config = TrainingPipelineConfig(cv_folds=5)

        assert config.discard_checkpoints is False

    def it_accepts_custom_discard_checkpoints(self) -> None:
        """Verify custom discard_checkpoints is stored."""
        config = TrainingPipelineConfig(cv_folds=5, discard_checkpoints=True)

        assert config.discard_checkpoints is True

    def it_has_none_n_jobs_by_default(self) -> None:
        """Verify n_jobs defaults to None."""
        config = TrainingPipelineConfig(cv_folds=5)

        assert config.n_jobs is None

    def it_accepts_custom_n_jobs(self) -> None:
        """Verify custom n_jobs is stored."""
        config = TrainingPipelineConfig(cv_folds=5, n_jobs=4)

        assert config.n_jobs == 4


class DescribeTrainingPipeline:
    """Tests for TrainingPipeline class."""

    @pytest.fixture
    def mock_task_generator(self) -> MagicMock:
        """Create mock task generator."""
        generator = MagicMock(spec=ExperimentTaskGenerator)
        generator.generate.return_value = [
            ExperimentTask(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=0,
            ),
        ]
        return generator

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        provider = MagicMock()
        provider.artifacts_exist.return_value = True
        provider.feature_context.return_value.__enter__ = MagicMock(
            return_value=("/path/X.joblib", "/path/y.joblib")
        )
        provider.feature_context.return_value.__exit__ = MagicMock(return_value=False)
        return provider

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock executor."""
        executor = MagicMock()
        executor.execute.return_value = ["result-0"]
        return executor

    @pytest.fixture
    def mock_persister(self) -> MagicMock:
        """Create mock persister."""
        return MagicMock(spec=ParquetCheckpointPersister)

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        provider = MagicMock()
        provider.get_checkpoint_path.return_value = Path("/checkpoints/test.parquet")
        return provider

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5)

    @pytest.fixture
    def pipeline(
        self,
        mock_task_generator: MagicMock,
        mock_data_provider: MagicMock,
        mock_executor: MagicMock,
        mock_persister: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
        config: TrainingPipelineConfig,
    ) -> TrainingPipeline:
        """Create training pipeline with mocked dependencies."""
        return TrainingPipeline(
            task_generator=mock_task_generator,
            data_provider=mock_data_provider,
            executor=mock_executor,
            persister=mock_persister,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
            config=config,
        )

    def it_stores_all_dependencies(
        self,
        pipeline: TrainingPipeline,
        mock_task_generator: MagicMock,
        mock_data_provider: MagicMock,
        mock_executor: MagicMock,
        mock_persister: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
        config: TrainingPipelineConfig,
    ) -> None:
        """Verify all dependencies are stored."""
        assert pipeline._task_generator is mock_task_generator
        assert pipeline._data_provider is mock_data_provider
        assert pipeline._executor is mock_executor
        assert pipeline._persister is mock_persister
        assert pipeline._consolidation_provider is mock_consolidation_provider
        assert pipeline._versioning_provider is mock_versioning_provider
        assert pipeline._experiment_runner_factory is mock_experiment_runner
        assert pipeline._config is config


class DescribeTrainingPipelineRun:
    """Tests for TrainingPipeline.run() method."""

    @pytest.fixture
    def mock_task_generator(self) -> MagicMock:
        """Create mock task generator."""
        generator = MagicMock(spec=ExperimentTaskGenerator)
        generator.generate.return_value = [
            ExperimentTask(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=0,
            ),
        ]
        return generator

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        provider = MagicMock()
        provider.artifacts_exist.return_value = True
        provider.feature_context.return_value.__enter__ = MagicMock(
            return_value=("/path/X.joblib", "/path/y.joblib")
        )
        provider.feature_context.return_value.__exit__ = MagicMock(return_value=False)
        return provider

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock executor."""
        executor = MagicMock()
        executor.execute.return_value = ["result-0"]
        return executor

    @pytest.fixture
    def mock_persister(self) -> MagicMock:
        """Create mock persister."""
        return MagicMock(spec=ParquetCheckpointPersister)

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5)

    @pytest.fixture
    def pipeline(
        self,
        mock_task_generator: MagicMock,
        mock_data_provider: MagicMock,
        mock_executor: MagicMock,
        mock_persister: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
        config: TrainingPipelineConfig,
    ) -> TrainingPipeline:
        """Create training pipeline with mocked dependencies."""
        return TrainingPipeline(
            task_generator=mock_task_generator,
            data_provider=mock_data_provider,
            executor=mock_executor,
            persister=mock_persister,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
            config=config,
        )

    def it_checks_data_availability(
        self,
        pipeline: TrainingPipeline,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verify data availability is checked."""
        pipeline.run(dataset=Dataset.TAIWAN_CREDIT)

        mock_data_provider.artifacts_exist.assert_called_once_with(Dataset.TAIWAN_CREDIT)

    def it_returns_empty_list_when_data_not_available(
        self,
        pipeline: TrainingPipeline,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verify empty list is returned when data unavailable."""
        mock_data_provider.artifacts_exist.return_value = False

        result = pipeline.run(dataset=Dataset.TAIWAN_CREDIT)

        assert result == []

    def it_generates_tasks_for_dataset(
        self,
        pipeline: TrainingPipeline,
        mock_task_generator: MagicMock,
    ) -> None:
        """Verify tasks are generated for dataset."""
        pipeline.run(dataset=Dataset.TAIWAN_CREDIT)

        mock_task_generator.generate.assert_called_once()
        args = mock_task_generator.generate.call_args[0]
        assert args[0] == [Dataset.TAIWAN_CREDIT]

    def it_executes_generated_tasks(
        self,
        pipeline: TrainingPipeline,
        mock_executor: MagicMock,
    ) -> None:
        """Verify generated tasks are executed."""
        pipeline.run(dataset=Dataset.TAIWAN_CREDIT)

        mock_executor.execute.assert_called_once()

    def it_consolidates_results_after_execution(
        self,
        pipeline: TrainingPipeline,
        mock_persister: MagicMock,
    ) -> None:
        """Verify results are consolidated after execution."""
        pipeline.run(dataset=Dataset.TAIWAN_CREDIT)

        mock_persister.consolidate.assert_called_once_with(Dataset.TAIWAN_CREDIT)

    def it_returns_executor_results(
        self,
        pipeline: TrainingPipeline,
        mock_executor: MagicMock,
    ) -> None:
        """Verify executor results are returned."""
        mock_executor.execute.return_value = ["task-0", "task-1"]

        result = pipeline.run(dataset=Dataset.TAIWAN_CREDIT)

        assert result == ["task-0", "task-1"]


class DescribeTrainingPipelineRunAll:
    """Tests for TrainingPipeline.run_all() method."""

    @pytest.fixture
    def mock_task_generator(self) -> MagicMock:
        """Create mock task generator."""
        generator = MagicMock(spec=ExperimentTaskGenerator)
        generator.generate.return_value = [
            ExperimentTask(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=0,
            ),
        ]
        return generator

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        provider = MagicMock()
        provider.artifacts_exist.return_value = True
        provider.feature_context.return_value.__enter__ = MagicMock(
            return_value=("/path/X.joblib", "/path/y.joblib")
        )
        provider.feature_context.return_value.__exit__ = MagicMock(return_value=False)
        provider.get_dataset_size_gb.return_value = 1.5
        return provider

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock executor."""
        executor = MagicMock()
        executor.execute.return_value = ["result-0"]
        return executor

    @pytest.fixture
    def mock_persister(self) -> MagicMock:
        """Create mock persister."""
        return MagicMock(spec=ParquetCheckpointPersister)

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5)

    @pytest.fixture
    def pipeline(
        self,
        mock_task_generator: MagicMock,
        mock_data_provider: MagicMock,
        mock_executor: MagicMock,
        mock_persister: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
        config: TrainingPipelineConfig,
    ) -> TrainingPipeline:
        """Create training pipeline with mocked dependencies."""
        return TrainingPipeline(
            task_generator=mock_task_generator,
            data_provider=mock_data_provider,
            executor=mock_executor,
            persister=mock_persister,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
            config=config,
        )

    def it_processes_multiple_datasets(
        self,
        pipeline: TrainingPipeline,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verify multiple datasets are processed."""
        datasets = [Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB]

        pipeline.run_all(datasets=datasets)

        # Should check artifacts for each dataset
        assert mock_data_provider.artifacts_exist.call_count == 2

    def it_returns_dict_with_results_per_dataset(
        self,
        pipeline: TrainingPipeline,
        mock_executor: MagicMock,
    ) -> None:
        """Verify returns dict mapping dataset IDs to results."""
        datasets = [Dataset.TAIWAN_CREDIT]
        mock_executor.execute.return_value = ["task-0"]

        result = pipeline.run_all(datasets=datasets)

        assert isinstance(result, dict)
        assert Dataset.TAIWAN_CREDIT.id in result
        assert result[Dataset.TAIWAN_CREDIT.id] == ["task-0"]

    def it_uses_compute_jobs_fn_when_provided(
        self,
        pipeline: TrainingPipeline,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verify compute_jobs_fn is called when provided."""
        datasets = [Dataset.TAIWAN_CREDIT]
        compute_fn = MagicMock(return_value=4)
        mock_data_provider.get_dataset_size_gb.return_value = 2.5

        pipeline.run_all(datasets=datasets, compute_jobs_fn=compute_fn)

        compute_fn.assert_called_once_with(2.5)


class DescribeTrainingPipelineRunSingle:
    """Tests for TrainingPipeline.run_single() method."""

    @pytest.fixture
    def mock_task_generator(self) -> MagicMock:
        """Create mock task generator."""
        return MagicMock(spec=ExperimentTaskGenerator)

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        provider = MagicMock()
        provider.artifacts_exist.return_value = True
        provider.feature_context.return_value.__enter__ = MagicMock(
            return_value=("/path/X.joblib", "/path/y.joblib")
        )
        provider.feature_context.return_value.__exit__ = MagicMock(return_value=False)
        return provider

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock executor."""
        executor = MagicMock()
        executor.execute.return_value = ["result-0"]
        return executor

    @pytest.fixture
    def mock_persister(self) -> MagicMock:
        """Create mock persister."""
        return MagicMock(spec=ParquetCheckpointPersister)

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5)

    @pytest.fixture
    def pipeline(
        self,
        mock_task_generator: MagicMock,
        mock_data_provider: MagicMock,
        mock_executor: MagicMock,
        mock_persister: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
        config: TrainingPipelineConfig,
    ) -> TrainingPipeline:
        """Create training pipeline with mocked dependencies."""
        return TrainingPipeline(
            task_generator=mock_task_generator,
            data_provider=mock_data_provider,
            executor=mock_executor,
            persister=mock_persister,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
            config=config,
        )

    @pytest.fixture
    def sample_task(self) -> ExperimentTask:
        """Create sample experiment task."""
        return ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=0,
        )

    def it_checks_data_availability(
        self,
        pipeline: TrainingPipeline,
        mock_data_provider: MagicMock,
        sample_task: ExperimentTask,
    ) -> None:
        """Verify data availability is checked."""
        pipeline.run_single(task=sample_task)

        mock_data_provider.artifacts_exist.assert_called_once_with(Dataset.TAIWAN_CREDIT)

    def it_returns_none_when_data_unavailable(
        self,
        pipeline: TrainingPipeline,
        mock_data_provider: MagicMock,
        sample_task: ExperimentTask,
    ) -> None:
        """Verify None is returned when data unavailable."""
        mock_data_provider.artifacts_exist.return_value = False

        result = pipeline.run_single(task=sample_task)

        assert result is None

    def it_executes_single_task(
        self,
        pipeline: TrainingPipeline,
        mock_executor: MagicMock,
        sample_task: ExperimentTask,
    ) -> None:
        """Verify single task is executed."""
        pipeline.run_single(task=sample_task)

        mock_executor.execute.assert_called_once()
        args = mock_executor.execute.call_args
        assert args[1]["tasks"] == [sample_task]

    def it_returns_task_result(
        self,
        pipeline: TrainingPipeline,
        mock_executor: MagicMock,
        sample_task: ExperimentTask,
    ) -> None:
        """Verify task result is returned."""
        mock_executor.execute.return_value = ["task-result"]

        result = pipeline.run_single(task=sample_task)

        assert result == "task-result"


class DescribeTrainingPipelineConsolidate:
    """Tests for TrainingPipeline.consolidate() method."""

    @pytest.fixture
    def mock_task_generator(self) -> MagicMock:
        """Create mock task generator."""
        return MagicMock(spec=ExperimentTaskGenerator)

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        return MagicMock()

    @pytest.fixture
    def mock_executor(self) -> MagicMock:
        """Create mock executor."""
        return MagicMock()

    @pytest.fixture
    def mock_persister(self) -> MagicMock:
        """Create mock persister."""
        persister = MagicMock(spec=ParquetCheckpointPersister)
        persister.consolidate.return_value = Path("/output/results.parquet")
        return persister

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5)

    @pytest.fixture
    def pipeline(
        self,
        mock_task_generator: MagicMock,
        mock_data_provider: MagicMock,
        mock_executor: MagicMock,
        mock_persister: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
        config: TrainingPipelineConfig,
    ) -> TrainingPipeline:
        """Create training pipeline with mocked dependencies."""
        return TrainingPipeline(
            task_generator=mock_task_generator,
            data_provider=mock_data_provider,
            executor=mock_executor,
            persister=mock_persister,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
            config=config,
        )

    def it_calls_persister_consolidate(
        self,
        pipeline: TrainingPipeline,
        mock_persister: MagicMock,
    ) -> None:
        """Verify persister consolidate is called."""
        pipeline.consolidate(Dataset.TAIWAN_CREDIT)

        mock_persister.consolidate.assert_called_once_with(Dataset.TAIWAN_CREDIT)

    def it_returns_persister_result(
        self,
        pipeline: TrainingPipeline,
        mock_persister: MagicMock,
    ) -> None:
        """Verify persister result is returned."""
        expected_path = Path("/output/results.parquet")
        mock_persister.consolidate.return_value = expected_path

        result = pipeline.consolidate(Dataset.TAIWAN_CREDIT)

        assert result == expected_path


class DescribeTrainingPipelineFactory:
    """Tests for TrainingPipelineFactory class."""

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        return MagicMock()

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def factory(
        self,
        storage: StorageService,
        mock_data_provider: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
    ) -> TrainingPipelineFactory:
        """Create factory with mocked dependencies."""
        return TrainingPipelineFactory(
            storage=storage,
            data_provider=mock_data_provider,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
        )

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5, num_seeds=10)

    def it_stores_dependencies(
        self,
        factory: TrainingPipelineFactory,
        storage: StorageService,
        mock_data_provider: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
    ) -> None:
        """Verify dependencies are stored."""
        assert factory._storage is storage
        assert factory._data_provider is mock_data_provider
        assert factory._consolidation_provider is mock_consolidation_provider
        assert factory._versioning_provider is mock_versioning_provider
        assert factory._experiment_runner_factory is mock_experiment_runner


class DescribeTrainingPipelineFactoryCreateParallel:
    """Tests for TrainingPipelineFactory.create_parallel_pipeline() method."""

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        return MagicMock()

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def factory(
        self,
        storage: StorageService,
        mock_data_provider: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
    ) -> TrainingPipelineFactory:
        """Create factory with mocked dependencies."""
        return TrainingPipelineFactory(
            storage=storage,
            data_provider=mock_data_provider,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
        )

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5, num_seeds=10)

    def it_returns_training_pipeline(
        self,
        factory: TrainingPipelineFactory,
        config: TrainingPipelineConfig,
    ) -> None:
        """Verify a TrainingPipeline is returned."""
        pipeline = factory.create_parallel_pipeline(config=config)

        assert isinstance(pipeline, TrainingPipeline)

    def it_uses_parallel_executor(
        self,
        factory: TrainingPipelineFactory,
        config: TrainingPipelineConfig,
    ) -> None:
        """Verify ParallelExecutor is used."""
        pipeline = factory.create_parallel_pipeline(config=config)

        assert isinstance(pipeline._executor, ParallelExecutor)

    def it_accepts_custom_n_jobs(
        self,
        factory: TrainingPipelineFactory,
        config: TrainingPipelineConfig,
    ) -> None:
        """Verify custom n_jobs is passed to executor."""
        pipeline = factory.create_parallel_pipeline(config=config, n_jobs=4)

        assert pipeline._executor.n_jobs == 4  # type: ignore[union-attr, attr-defined]


class DescribeTrainingPipelineFactoryCreateSequential:
    """Tests for TrainingPipelineFactory.create_sequential_pipeline() method."""

    @pytest.fixture
    def mock_data_provider(self) -> MagicMock:
        """Create mock data provider."""
        return MagicMock()

    @pytest.fixture
    def mock_consolidation_provider(self) -> MagicMock:
        """Create mock consolidation provider."""
        return MagicMock()

    @pytest.fixture
    def mock_versioning_provider(self) -> MagicMock:
        """Create mock versioning provider."""
        return MagicMock()

    @pytest.fixture
    def mock_experiment_runner(self) -> MagicMock:
        """Create mock experiment runner."""
        return MagicMock()

    @pytest.fixture
    def factory(
        self,
        storage: StorageService,
        mock_data_provider: MagicMock,
        mock_consolidation_provider: MagicMock,
        mock_versioning_provider: MagicMock,
        mock_experiment_runner: MagicMock,
    ) -> TrainingPipelineFactory:
        """Create factory with mocked dependencies."""
        return TrainingPipelineFactory(
            storage=storage,
            data_provider=mock_data_provider,
            consolidation_provider=mock_consolidation_provider,
            versioning_provider=mock_versioning_provider,
            experiment_runner_factory=mock_experiment_runner,
        )

    @pytest.fixture
    def config(self) -> TrainingPipelineConfig:
        """Create pipeline config."""
        return TrainingPipelineConfig(cv_folds=5, num_seeds=10)

    def it_returns_training_pipeline(
        self,
        factory: TrainingPipelineFactory,
        config: TrainingPipelineConfig,
    ) -> None:
        """Verify a TrainingPipeline is returned."""
        pipeline = factory.create_sequential_pipeline(config=config)

        assert isinstance(pipeline, TrainingPipeline)

    def it_uses_sequential_executor(
        self,
        factory: TrainingPipelineFactory,
        config: TrainingPipelineConfig,
    ) -> None:
        """Verify SequentialExecutor is used."""
        pipeline = factory.create_sequential_pipeline(config=config)

        assert isinstance(pipeline._executor, SequentialExecutor)
