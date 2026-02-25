"""Tests for experiment_executor service."""

from unittest.mock import MagicMock

import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import ExperimentCombination
from experiments.services.experiment_executor import (
    ExperimentConfig,
    ExperimentExecutor,
    ExperimentParams,
)


@pytest.fixture
def mock_training_pipeline_factory() -> MagicMock:
    """Mock TrainingPipelineFactory."""
    factory = MagicMock()
    factory.create_pipeline.return_value = MagicMock()
    return factory


@pytest.fixture
def mock_pipeline_executor() -> MagicMock:
    """Mock PipelineExecutor."""
    executor = MagicMock()
    return executor


@pytest.fixture
def mock_model_trainer() -> MagicMock:
    """Mock ModelTrainer."""
    return MagicMock()


@pytest.fixture
def mock_data_splitter() -> MagicMock:
    """Mock DataSplitter."""
    return MagicMock()


@pytest.fixture
def mock_training_data_loader() -> MagicMock:
    """Mock TrainingDataLoader."""
    return MagicMock()


@pytest.fixture
def mock_classifier_factory() -> MagicMock:
    """Mock ClassifierFactory."""
    return MagicMock()


@pytest.fixture
def mock_predictions_repository() -> MagicMock:
    """Mock ModelPredictionsRepository."""
    repository = MagicMock()
    repository.get_completed_combinations.return_value = set()
    return repository


@pytest.fixture
def mock_experiment_settings() -> MagicMock:
    """Mock ExperimentSettings."""
    settings = MagicMock()
    settings.num_seeds = 5
    return settings


@pytest.fixture
def mock_resource_settings() -> MagicMock:
    """Mock ResourceSettings."""
    settings = MagicMock()
    settings.use_gpu = False
    settings.n_jobs = 4
    settings.models_n_jobs = 2
    settings.sequential = False
    return settings


@pytest.fixture
def executor(
    mock_training_pipeline_factory: MagicMock,
    mock_pipeline_executor: MagicMock,
    mock_model_trainer: MagicMock,
    mock_data_splitter: MagicMock,
    mock_training_data_loader: MagicMock,
    mock_classifier_factory: MagicMock,
    mock_predictions_repository: MagicMock,
    mock_experiment_settings: MagicMock,
    mock_resource_settings: MagicMock,
) -> ExperimentExecutor:
    """ExperimentExecutor instance with mocked dependencies."""
    return ExperimentExecutor(
        training_pipeline_factory=mock_training_pipeline_factory,
        pipeline_executor=mock_pipeline_executor,
        model_trainer=mock_model_trainer,
        data_splitter=mock_data_splitter,
        training_data_loader=mock_training_data_loader,
        classifier_factory=mock_classifier_factory,
        predictions_repository=mock_predictions_repository,
        experiment_settings=mock_experiment_settings,
        resource_settings=mock_resource_settings,
    )


class DescribeExperimentExecutorInit:
    def it_stores_all_dependencies(
        self,
        mock_training_pipeline_factory: MagicMock,
        mock_pipeline_executor: MagicMock,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        mock_predictions_repository: MagicMock,
        mock_experiment_settings: MagicMock,
        mock_resource_settings: MagicMock,
    ) -> None:
        executor = ExperimentExecutor(
            training_pipeline_factory=mock_training_pipeline_factory,
            pipeline_executor=mock_pipeline_executor,
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=mock_predictions_repository,
            experiment_settings=mock_experiment_settings,
            resource_settings=mock_resource_settings,
        )

        assert executor._training_pipeline_factory is mock_training_pipeline_factory
        assert executor._pipeline_executor is mock_pipeline_executor
        assert executor._model_trainer is mock_model_trainer
        assert executor._data_splitter is mock_data_splitter
        assert executor._training_data_loader is mock_training_data_loader
        assert executor._classifier_factory is mock_classifier_factory
        assert executor._predictions_repository is mock_predictions_repository

    def it_creates_default_config_from_settings(
        self,
        executor: ExperimentExecutor,
        mock_experiment_settings: MagicMock,
        mock_resource_settings: MagicMock,
    ) -> None:
        assert executor._default_config["num_seeds"] == mock_experiment_settings.num_seeds
        assert executor._default_config["use_gpu"] == mock_resource_settings.use_gpu
        assert executor._default_config["n_jobs"] == mock_resource_settings.n_jobs
        assert executor._default_config["models_n_jobs"] == mock_resource_settings.models_n_jobs
        assert executor._default_config["sequential"] == mock_resource_settings.sequential


class DescribeExecuteExperiment:
    def it_schedules_pipelines_for_all_combinations(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[],
        )
        config: ExperimentConfig = {"num_seeds": 2}

        executor.execute_experiment(params, config)

        # Calculate expected: 1 dataset * 4 model types * techniques * 2 seeds
        # But CS_SVM is only valid for SVM, so:
        # SVM: 5 techniques * 2 seeds = 10
        # Other 3 models: 4 techniques each * 2 seeds = 24
        # Total: 10 + 24 = 34
        assert mock_pipeline_executor.schedule.call_count == 34

    def it_starts_pipeline_executor_with_observers(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(datasets=[Dataset.TAIWAN_CREDIT])

        executor.execute_experiment(params)

        mock_pipeline_executor.start.assert_called_once()
        call_kwargs = mock_pipeline_executor.start.call_args.kwargs
        assert "observers" in call_kwargs
        assert len(call_kwargs["observers"]) == 1

    def it_waits_for_pipeline_completion(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(datasets=[Dataset.TAIWAN_CREDIT])

        executor.execute_experiment(params)

        mock_pipeline_executor.wait.assert_called_once()

    def it_uses_n_jobs_from_config(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(datasets=[Dataset.TAIWAN_CREDIT])
        config: ExperimentConfig = {"n_jobs": 8}

        executor.execute_experiment(params, config)

        mock_pipeline_executor.start.assert_called_once()
        call_kwargs = mock_pipeline_executor.start.call_args.kwargs
        assert call_kwargs["max_workers"] == 8

    def it_uses_default_n_jobs_when_not_in_config(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
        mock_resource_settings: MagicMock,
    ) -> None:
        params = ExperimentParams(datasets=[Dataset.TAIWAN_CREDIT])

        executor.execute_experiment(params)

        mock_pipeline_executor.start.assert_called_once()
        call_kwargs = mock_pipeline_executor.start.call_args.kwargs
        assert call_kwargs["max_workers"] == mock_resource_settings.n_jobs

    def it_merges_config_with_defaults(
        self,
        executor: ExperimentExecutor,
        mock_resource_settings: MagicMock,
    ) -> None:
        config: ExperimentConfig = {"num_seeds": 10}

        # Access default config to verify merge behavior
        merged = executor._merge_with_default_config(config)

        assert merged["num_seeds"] == 10  # from config
        assert merged["n_jobs"] == mock_resource_settings.n_jobs  # from default
        assert merged["use_gpu"] == mock_resource_settings.use_gpu  # from default

    def it_skips_already_completed_combinations(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
        mock_predictions_repository: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2}

        # Mock 5 completed combinations for SVM (out of 12 total)
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 2),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 2),
            ExperimentCombination(
                Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.RANDOM_UNDER_SAMPLING, 1
            ),
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        executor.execute_experiment(params, config)

        # Only SVM with 5 techniques * 2 seeds = 10 total, minus 5 completed = 5 scheduled
        assert mock_pipeline_executor.schedule.call_count == 5

    def it_processes_multiple_datasets(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 1}

        executor.execute_experiment(params, config)

        # 2 datasets * 1 model (SVM) * 5 techniques * 1 seed = 10
        assert mock_pipeline_executor.schedule.call_count == 10

    def it_excludes_specified_models(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST],
        )
        config: ExperimentConfig = {"num_seeds": 1}

        executor.execute_experiment(params, config)

        # 1 dataset * 2 models (SVM + MLP) * techniques * 1 seed
        # SVM: 5 techniques, MLP: 4 techniques = 9
        assert mock_pipeline_executor.schedule.call_count == 9


class DescribeGetCompletedCount:
    def it_returns_count_from_repository(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-123"
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.CS_SVM, 1),
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        count = executor.get_completed_count(execution_id)

        assert count == 3
        mock_predictions_repository.get_completed_combinations.assert_called_once_with(
            execution_id
        )

    def it_returns_zero_when_no_completed_combinations(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-456"
        mock_predictions_repository.get_completed_combinations.return_value = set()

        count = executor.get_completed_count(execution_id)

        assert count == 0


class DescribeIsExecutionComplete:
    def it_returns_true_when_all_combinations_complete(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-789"
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2}

        # Mock all 10 combinations for SVM (5 techniques * 2 seeds)
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, tech, seed)
            for tech in Technique
            for seed in [1, 2]
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        result = executor.is_execution_complete(execution_id, params, config)

        assert result is True

    def it_returns_false_when_incomplete(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-101"
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2}

        # Mock only 5 completed combinations (out of 10 expected)
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 2),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 2),
            ExperimentCombination(
                Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.RANDOM_UNDER_SAMPLING, 1
            ),
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        result = executor.is_execution_complete(execution_id, params, config)

        assert result is False

    def it_uses_config_num_seeds(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-202"
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 3}

        # Mock all combinations for 3 seeds (5 techniques * 3 seeds = 15)
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, tech, seed)
            for tech in Technique
            for seed in [1, 2, 3]
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        result = executor.is_execution_complete(execution_id, params, config)

        assert result is True

    def it_uses_default_num_seeds_when_not_in_config(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
        mock_experiment_settings: MagicMock,
    ) -> None:
        execution_id = "exec-303"
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {}

        # Mock all combinations using default num_seeds from settings (5)
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, tech, seed)
            for tech in Technique
            for seed in range(1, mock_experiment_settings.num_seeds + 1)
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        result = executor.is_execution_complete(execution_id, params, config)

        assert result is True

    def it_calculates_expected_count_correctly_for_multiple_datasets(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-404"
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2}

        # Mock all combinations for 2 datasets (2 * 5 techniques * 2 seeds = 20)
        completed = {
            ExperimentCombination(dataset, ModelType.SVM, tech, seed)
            for dataset in [Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB]
            for tech in Technique
            for seed in [1, 2]
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        result = executor.is_execution_complete(execution_id, params, config)

        assert result is True

    def it_ignores_completed_combinations_from_other_datasets(
        self,
        executor: ExperimentExecutor,
        mock_predictions_repository: MagicMock,
    ) -> None:
        execution_id = "exec-505"
        params = ExperimentParams(
            datasets=[Dataset.LENDING_CLUB],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2}

        # Complete all expected combinations for a different dataset
        # (Taiwan Credit) and only a subset for Lending Club.
        completed = {
            ExperimentCombination(dataset, ModelType.SVM, tech, seed)
            for dataset in [Dataset.TAIWAN_CREDIT]
            for tech in Technique
            for seed in [1, 2]
        }
        completed.update(
            {
                ExperimentCombination(Dataset.LENDING_CLUB, ModelType.SVM, Technique.BASELINE, 1),
                ExperimentCombination(Dataset.LENDING_CLUB, ModelType.SVM, Technique.BASELINE, 2),
            }
        )

        mock_predictions_repository.get_completed_combinations.return_value = completed

        result = executor.is_execution_complete(execution_id, params, config)

        assert result is False


class DescribeIsValidCombination:
    def it_allows_cs_svm_for_svm_model(self) -> None:
        result = ExperimentExecutor._is_valid_combination(ModelType.SVM, Technique.CS_SVM)

        assert result is True

    def it_rejects_cs_svm_for_random_forest(self) -> None:
        result = ExperimentExecutor._is_valid_combination(
            ModelType.RANDOM_FOREST, Technique.CS_SVM
        )

        assert result is False

    def it_rejects_cs_svm_for_xgboost(self) -> None:
        result = ExperimentExecutor._is_valid_combination(ModelType.XGBOOST, Technique.CS_SVM)

        assert result is False

    def it_rejects_cs_svm_for_mlp(self) -> None:
        result = ExperimentExecutor._is_valid_combination(ModelType.MLP, Technique.CS_SVM)

        assert result is False

    def it_allows_all_other_techniques_for_all_models(self) -> None:
        non_cs_svm_techniques = [
            Technique.BASELINE,
            Technique.SMOTE,
            Technique.RANDOM_UNDER_SAMPLING,
            Technique.SMOTE_TOMEK,
        ]

        for model_type in ModelType:
            for technique in non_cs_svm_techniques:
                result = ExperimentExecutor._is_valid_combination(model_type, technique)
                assert result is True, f"Failed for {model_type} + {technique}"


class DescribeExperimentParamsValidation:
    def it_rejects_excluding_all_models(self) -> None:
        with pytest.raises(Exception) as exc_info:
            ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT],
                excluded_models=list(ModelType),
            )

        assert "at least one model type" in str(exc_info.value).lower()

    def it_allows_excluding_some_models(self) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST],
        )

        assert ModelType.RANDOM_FOREST in params.excluded_models
        assert ModelType.XGBOOST in params.excluded_models
        assert ModelType.SVM not in params.excluded_models
        assert ModelType.MLP not in params.excluded_models

    def it_generates_execution_id_when_not_provided(self) -> None:
        params = ExperimentParams(datasets=[Dataset.TAIWAN_CREDIT])

        assert params.execution_id is not None
        assert isinstance(params.execution_id, str)
        assert len(params.execution_id) > 0

    def it_uses_provided_execution_id(self) -> None:
        exec_id = "custom-exec-id-123"
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            execution_id=exec_id,
        )

        assert params.execution_id == exec_id


class DescribeSeedGeneration:
    def it_generates_seeds_from_one_to_num_seeds(
        self,
        executor: ExperimentExecutor,
    ) -> None:
        seeds = list(executor._generate_seeds(5))

        assert seeds == [1, 2, 3, 4, 5]

    def it_generates_single_seed_when_num_seeds_is_one(
        self,
        executor: ExperimentExecutor,
    ) -> None:
        seeds = list(executor._generate_seeds(1))

        assert seeds == [1]

    def it_generates_empty_list_when_num_seeds_is_zero(
        self,
        executor: ExperimentExecutor,
    ) -> None:
        seeds = list(executor._generate_seeds(0))

        assert seeds == []


class DescribeSequentialExecution:
    def it_executes_pipelines_sequentially_when_config_is_set(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2, "sequential": True}

        executor.execute_experiment(params, config)

        # SVM: 5 techniques * 2 seeds = 10 combinations
        # Each should trigger schedule, start, wait, reset
        assert mock_pipeline_executor.schedule.call_count == 10
        assert mock_pipeline_executor.start.call_count == 10
        assert mock_pipeline_executor.wait.call_count == 10
        assert mock_pipeline_executor.reset.call_count == 10

    def it_does_not_use_schedule_for_parallel_path_in_sequential_mode(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        """In sequential mode, _schedule_pipelines (bulk) is NOT called;
        instead, individual schedule/start/wait/reset cycles are used."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 1, "sequential": True}

        executor.execute_experiment(params, config)

        # In sequential mode, start is called once per combination (not once overall)
        assert mock_pipeline_executor.start.call_count == 5
        # And each start is paired with a wait and reset
        assert mock_pipeline_executor.wait.call_count == 5
        assert mock_pipeline_executor.reset.call_count == 5

    def it_skips_completed_combinations_in_sequential_mode(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
        mock_predictions_repository: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 2, "sequential": True}

        # Mock 5 completed combinations for SVM
        completed = {
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.BASELINE, 2),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 1),
            ExperimentCombination(Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.SMOTE, 2),
            ExperimentCombination(
                Dataset.TAIWAN_CREDIT, ModelType.SVM, Technique.RANDOM_UNDER_SAMPLING, 1
            ),
        }
        mock_predictions_repository.get_completed_combinations.return_value = completed

        executor.execute_experiment(params, config)

        # 10 total - 5 completed = 5 remaining
        assert mock_pipeline_executor.schedule.call_count == 5
        assert mock_pipeline_executor.start.call_count == 5
        assert mock_pipeline_executor.wait.call_count == 5
        assert mock_pipeline_executor.reset.call_count == 5

    def it_uses_parallel_by_default(
        self,
        executor: ExperimentExecutor,
        mock_pipeline_executor: MagicMock,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.MLP],
        )
        config: ExperimentConfig = {"num_seeds": 1}

        executor.execute_experiment(params, config)

        # Parallel mode: schedule is called for all, then start once and wait once
        assert mock_pipeline_executor.schedule.call_count == 5
        assert mock_pipeline_executor.start.call_count == 1
        assert mock_pipeline_executor.wait.call_count == 1
        # reset is NOT called in parallel mode
        assert mock_pipeline_executor.reset.call_count == 0
