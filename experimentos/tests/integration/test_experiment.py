"""Integration tests for experiment execution.

These tests verify the experiment execution orchestration logic,
using mocks for the training components to ensure fast execution.
The actual training is tested separately in test_training.py.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.config.settings import ExperimentSettings, ResourceSettings
from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import (
    ExperimentCombination,
    ModelPredictions,
    ModelPredictionsRepository,
    RawPredictions,
)
from experiments.lib.pipelines import (
    PipelineExecutionResult,
    PipelineExecutor,
    PipelineStatus,
)
from experiments.pipelines.training.factory import TrainingPipelineFactory
from experiments.pipelines.training.pipeline import (
    TrainingPipelineState,
)
from experiments.services.experiment_executor import (
    ExperimentConfig,
    ExperimentExecutor,
    ExperimentParams,
)

# =============================================================================
# Fake predictions repository for testing
# =============================================================================


class FakePredictionsRepository(ModelPredictionsRepository):
    """In-memory fake predictions repository for testing."""

    def __init__(self):
        self._predictions: dict[str, dict[ExperimentCombination, RawPredictions]] = {}

    def save_predictions(
        self,
        *,
        execution_id: str,
        seed: int,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        predictions: RawPredictions,
    ) -> None:
        """Save predictions to in-memory storage."""
        if execution_id not in self._predictions:
            self._predictions[execution_id] = {}

        combination = ExperimentCombination(
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            seed=seed,
        )
        self._predictions[execution_id][combination] = predictions

    def get_completed_combinations(self, execution_id: str) -> set[ExperimentCombination]:
        """Get all completed combinations for an execution."""
        if execution_id not in self._predictions:
            return set()
        return set(self._predictions[execution_id].keys())

    def get_latest_execution_id(self, datasets: list[Dataset] | None = None) -> str | None:
        """Get the latest execution ID, optionally filtered by datasets."""
        if not self._predictions:
            return None

        # Filter executions by datasets if specified
        matching_executions: set[str] = set()
        for execution_id, combinations in self._predictions.items():
            if datasets is None:
                matching_executions.add(execution_id)
            else:
                # Check if this execution has any of the requested datasets
                execution_datasets = {combo.dataset for combo in combinations}
                if any(dataset in execution_datasets for dataset in datasets):
                    matching_executions.add(execution_id)

        if not matching_executions:
            return None

        # Return the max (latest) execution ID (UUID7 is time-sortable)
        return max(matching_executions)

    def get_predictions(
        self, execution_id: str, combination: ExperimentCombination
    ) -> RawPredictions | None:
        """Get predictions for a specific combination."""
        if execution_id not in self._predictions:
            return None
        return self._predictions[execution_id].get(combination)

    def get_all_predictions(
        self, execution_id: str
    ) -> dict[ExperimentCombination, RawPredictions]:
        """Get all predictions for an execution."""
        return self._predictions.get(execution_id, {})

    def clear(self) -> None:
        """Clear all stored predictions."""
        self._predictions.clear()

    def get_latest_predictions_for_experiment(
        self, dataset: Dataset
    ) -> Iterator[ModelPredictions] | None:
        """Not implemented for fake repository."""
        return None

    def get_predictions_for_execution(self, dataset, execution_id):
        """Not implemented for fake repository."""
        return None


@pytest.fixture
def predictions_repository() -> FakePredictionsRepository:
    """Fixture providing a fake in-memory predictions repository."""
    return FakePredictionsRepository()


# =============================================================================
# Mock fixtures for training components
# =============================================================================


@pytest.fixture
def mock_model_trainer() -> MagicMock:
    """Fixture providing a mock model trainer."""
    return MagicMock()


@pytest.fixture
def mock_data_splitter() -> MagicMock:
    """Fixture providing a mock data splitter."""
    return MagicMock()


@pytest.fixture
def mock_training_data_loader() -> MagicMock:
    """Fixture providing a mock training data loader."""
    return MagicMock()


@pytest.fixture
def mock_classifier_factory() -> MagicMock:
    """Fixture providing a mock classifier factory."""
    return MagicMock()


@pytest.fixture
def training_pipeline_factory() -> TrainingPipelineFactory:
    """Fixture providing a real training pipeline factory."""
    return TrainingPipelineFactory()


@pytest.fixture
def experiment_settings() -> ExperimentSettings:
    """Fixture providing minimal experiment settings for fast tests."""
    return ExperimentSettings(
        cv_folds=2,
        num_seeds=2,
        cost_grids=[{"C": 1.0}],
    )


@pytest.fixture
def resource_settings() -> ResourceSettings:
    """Fixture providing resource settings for tests."""
    return ResourceSettings(
        use_gpu=False,
        n_jobs=1,
        models_n_jobs=1,
        sequential=False,
    )


def _create_mock_pipeline_executor(
    predictions_repository: FakePredictionsRepository,
) -> PipelineExecutor:
    """Create a mock pipeline executor that simulates successful execution."""

    class MockPipelineExecutor:
        """Mock executor that stores scheduled pipelines and executes them with fake results."""

        def __init__(self) -> None:
            self._scheduled: list[tuple[Any, Any, Any]] = []
            self._execution_id: str | None = None

        def schedule(
            self,
            pipeline: Any,
            initial_state: Any,
            context: Any,
        ) -> None:
            """Store scheduled pipeline for later execution."""
            self._scheduled.append((pipeline, initial_state, context))

        def start(
            self,
            observers: Any = None,
            max_workers: int = 1,
        ) -> None:
            """Execute all scheduled pipelines with fake results."""
            self._observers = observers or set()

            for pipeline, initial_state, context in self._scheduled:
                # Create fake predictions
                fake_predictions = RawPredictions(
                    target=np.array([0, 1, 0, 1]),
                    prediction=np.array([0, 1, 0, 1]),
                )

                # Create fake result state
                final_state: TrainingPipelineState = {
                    "predictions": fake_predictions,
                }

                # Create mock result
                result = MagicMock(spec=PipelineExecutionResult)
                result.context = context
                result.final_state = final_state
                result.succeeded.return_value = True
                result.status = PipelineStatus.COMPLETED

                # Notify observers
                for observer in self._observers:
                    observer.on_pipeline_finish(pipeline, result)

            self._scheduled.clear()

        def wait(self) -> list[Any]:
            """No-op for mock."""
            return []

        def reset(self) -> None:
            """Reset for reuse in sequential mode."""
            self._scheduled.clear()

    return MockPipelineExecutor()  # type: ignore[return-value]


@pytest.fixture
def mock_pipeline_executor(predictions_repository: FakePredictionsRepository):
    """Fixture providing a mock pipeline executor."""
    return _create_mock_pipeline_executor(predictions_repository)


@pytest.fixture
def experiment_executor(
    training_pipeline_factory: TrainingPipelineFactory,
    mock_pipeline_executor,
    mock_model_trainer: MagicMock,
    mock_data_splitter: MagicMock,
    mock_training_data_loader: MagicMock,
    mock_classifier_factory: MagicMock,
    predictions_repository: ModelPredictionsRepository,
    experiment_settings: ExperimentSettings,
    resource_settings: ResourceSettings,
) -> ExperimentExecutor:
    """Fixture providing a fully wired ExperimentExecutor with mocks."""
    return ExperimentExecutor(
        training_pipeline_factory=training_pipeline_factory,
        pipeline_executor=mock_pipeline_executor,
        model_trainer=mock_model_trainer,
        data_splitter=mock_data_splitter,
        training_data_loader=mock_training_data_loader,
        classifier_factory=mock_classifier_factory,
        predictions_repository=predictions_repository,
        experiment_settings=experiment_settings,
        resource_settings=resource_settings,
    )


# =============================================================================
# Integration tests for ExperimentParams
# =============================================================================


class DescribeExperimentParams:
    """Tests for ExperimentParams validation."""

    def it_generates_execution_id_by_default(self) -> None:
        params = ExperimentParams()

        assert params.execution_id is not None
        assert len(params.execution_id) > 0

    def it_uses_all_datasets_by_default(self) -> None:
        params = ExperimentParams()

        assert params.datasets == list(Dataset)

    def it_accepts_subset_of_datasets(self) -> None:
        params = ExperimentParams(datasets=[Dataset.TAIWAN_CREDIT])

        assert params.datasets == [Dataset.TAIWAN_CREDIT]

    def it_accepts_excluded_models(self) -> None:
        params = ExperimentParams(
            excluded_models=[ModelType.SVM, ModelType.MLP],
        )

        assert ModelType.SVM in params.excluded_models
        assert ModelType.MLP in params.excluded_models

    def it_rejects_excluding_all_models(self) -> None:
        with pytest.raises(ValueError, match="At least one model type must be included"):
            ExperimentParams(excluded_models=list(ModelType))


# =============================================================================
# Integration tests for ExperimentExecutor
# =============================================================================


class DescribeExperimentExecutor:
    """Integration tests for the ExperimentExecutor service."""

    class DescribeExecuteExperiment:
        """Tests for the execute_experiment method."""

        def it_executes_experiment_with_single_model_and_technique(
            self,
            experiment_executor: ExperimentExecutor,
            predictions_repository: FakePredictionsRepository,
        ) -> None:
            """Test a minimal experiment with one model type."""
            params = ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT],
                excluded_models=[
                    ModelType.SVM,
                    ModelType.XGBOOST,
                    ModelType.MLP,
                ],  # Only Random Forest
            )
            config: ExperimentConfig = {
                "num_seeds": 1,
                "n_jobs": 1,
                "models_n_jobs": 1,
                "use_gpu": False,
            }

            experiment_executor.execute_experiment(params, config)

            # Should have trained for each valid technique with 1 seed
            # Valid techniques for RF: BASELINE, SMOTE, RANDOM_UNDER_SAMPLING, SMOTE_TOMEK
            # CS_SVM is only for SVM, so should be excluded
            completed = predictions_repository.get_completed_combinations(params.execution_id)
            assert len(completed) == 4  # 4 techniques * 1 seed

        def it_saves_predictions_for_each_combination(
            self,
            experiment_executor: ExperimentExecutor,
            predictions_repository: FakePredictionsRepository,
        ) -> None:
            """Test that predictions are saved for each trained model."""
            params = ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT],
                excluded_models=[
                    ModelType.SVM,
                    ModelType.XGBOOST,
                    ModelType.MLP,
                ],
            )
            config: ExperimentConfig = {
                "num_seeds": 1,
                "n_jobs": 1,
                "models_n_jobs": 1,
                "use_gpu": False,
            }

            experiment_executor.execute_experiment(params, config)

            # Check that predictions contain actual data
            completed = predictions_repository.get_completed_combinations(params.execution_id)
            for combination in completed:
                preds = predictions_repository.get_predictions(params.execution_id, combination)
                assert preds is not None
                assert len(preds.prediction) > 0
                assert len(preds.target) > 0
                assert len(preds.prediction) == len(preds.target)

        def it_trains_multiple_seeds(
            self,
            experiment_executor: ExperimentExecutor,
            predictions_repository: FakePredictionsRepository,
        ) -> None:
            """Test that multiple seeds are trained."""
            params = ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT],
                excluded_models=[
                    ModelType.SVM,
                    ModelType.XGBOOST,
                    ModelType.MLP,
                ],
            )
            config: ExperimentConfig = {
                "num_seeds": 2,
                "n_jobs": 1,
                "models_n_jobs": 1,
                "use_gpu": False,
            }

            experiment_executor.execute_experiment(params, config)

            # 4 techniques * 2 seeds = 8 combinations
            completed = predictions_repository.get_completed_combinations(params.execution_id)
            assert len(completed) == 8

            # Check both seeds are present
            seeds = {c.seed for c in completed}
            assert seeds == {1, 2}

        def it_trains_multiple_datasets(
            self,
            experiment_executor: ExperimentExecutor,
            predictions_repository: FakePredictionsRepository,
        ) -> None:
            """Test that multiple datasets are processed."""
            params = ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB],
                excluded_models=[
                    ModelType.SVM,
                    ModelType.XGBOOST,
                    ModelType.MLP,
                ],
            )
            config: ExperimentConfig = {
                "num_seeds": 1,
                "n_jobs": 1,
                "models_n_jobs": 1,
                "use_gpu": False,
            }

            experiment_executor.execute_experiment(params, config)

            # 2 datasets * 4 techniques * 1 seed = 8 combinations
            completed = predictions_repository.get_completed_combinations(params.execution_id)
            assert len(completed) == 8

            # Check both datasets are present
            datasets = {c.dataset for c in completed}
            assert datasets == {Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB}

        def it_includes_cs_svm_only_for_svm_model(
            self,
            experiment_executor: ExperimentExecutor,
            predictions_repository: FakePredictionsRepository,
        ) -> None:
            """Test that CS_SVM technique is only used with SVM model type."""
            params = ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT],
                excluded_models=[
                    ModelType.RANDOM_FOREST,
                    ModelType.XGBOOST,
                    ModelType.MLP,
                ],  # Only SVM
            )
            config: ExperimentConfig = {
                "num_seeds": 1,
                "n_jobs": 1,
                "models_n_jobs": 1,
                "use_gpu": False,
            }

            experiment_executor.execute_experiment(params, config)

            # SVM has all 5 techniques including CS_SVM
            completed = predictions_repository.get_completed_combinations(params.execution_id)
            techniques = {c.technique for c in completed}

            assert Technique.CS_SVM in techniques
            assert len(techniques) == 5  # All techniques valid for SVM

        def it_trains_all_model_types_when_none_excluded(
            self,
            experiment_executor: ExperimentExecutor,
            predictions_repository: FakePredictionsRepository,
        ) -> None:
            """Test that all model types are trained when none are excluded."""
            params = ExperimentParams(
                datasets=[Dataset.TAIWAN_CREDIT],
                excluded_models=[],
            )
            config: ExperimentConfig = {
                "num_seeds": 1,
                "n_jobs": 1,
                "models_n_jobs": 1,
                "use_gpu": False,
            }

            experiment_executor.execute_experiment(params, config)

            completed = predictions_repository.get_completed_combinations(params.execution_id)
            model_types = {c.model_type for c in completed}

            assert model_types == set(ModelType)


class DescribeExperimentResumption:
    """Integration tests for experiment resumption (continuation)."""

    def it_skips_already_completed_combinations(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test that completed combinations are skipped on resumption."""
        execution_id = "test-resumption-id"
        predictions_repository = FakePredictionsRepository()

        # First execution
        executor1 = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=_create_mock_pipeline_executor(predictions_repository),
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        params = ExperimentParams(
            execution_id=execution_id,
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 2,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        executor1.execute_experiment(params, config)
        first_run_count = len(predictions_repository.get_completed_combinations(execution_id))
        assert first_run_count == 8  # 4 techniques * 2 seeds

        # Track how many pipelines are scheduled in second run
        mock_executor2 = _create_mock_pipeline_executor(predictions_repository)
        original_schedule = mock_executor2.schedule
        scheduled_count: list[int] = []

        def tracking_schedule(pipeline: Any, initial_state: Any, context: Any) -> None:
            scheduled_count.append(1)
            original_schedule(pipeline, initial_state, context)

        mock_executor2.schedule = tracking_schedule  # type: ignore[method-assign]

        # Second execution with same execution_id
        executor2 = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=mock_executor2,
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        executor2.execute_experiment(params, config)

        # No new pipelines should be scheduled (all already completed)
        assert len(scheduled_count) == 0

    def it_completes_remaining_combinations_on_resumption(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test that remaining combinations are completed on resumption."""
        execution_id = "test-partial-resumption"
        predictions_repository = FakePredictionsRepository()

        # Pre-populate some completed combinations
        pre_completed = ExperimentCombination(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=1,
        )
        predictions_repository.save_predictions(
            execution_id=execution_id,
            seed=pre_completed.seed,
            dataset=pre_completed.dataset,
            model_type=pre_completed.model_type,
            technique=pre_completed.technique,
            predictions=RawPredictions(
                target=np.array([0, 1]),
                prediction=np.array([0, 1]),
            ),
        )

        executor = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=_create_mock_pipeline_executor(predictions_repository),
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        params = ExperimentParams(
            execution_id=execution_id,
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        executor.execute_experiment(params, config)

        # Should have completed the remaining 3 techniques (BASELINE already done)
        completed = predictions_repository.get_completed_combinations(execution_id)
        assert len(completed) == 4  # All 4 techniques completed

        # The pre-completed one should still be there
        assert pre_completed in completed

    def it_simulates_auto_resume_workflow(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test the complete auto-resume workflow: find latest, check complete, resume if needed.

        This simulates what the CLI does: get latest execution ID, check if complete,
        and resume automatically if not complete.
        """
        predictions_repository = FakePredictionsRepository()

        # Step 1: First execution - simulate partial completion
        executor1 = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=_create_mock_pipeline_executor(predictions_repository),
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        # Pre-populate with partial results (simulating interrupted execution)
        first_exec_id = "01943abc-1234-7000-8000-0123456789ab"
        for seed in [1]:  # Only first seed completed
            for technique in [Technique.BASELINE, Technique.SMOTE]:  # Only 2 techniques
                predictions_repository.save_predictions(
                    execution_id=first_exec_id,
                    seed=seed,
                    dataset=Dataset.TAIWAN_CREDIT,
                    model_type=ModelType.RANDOM_FOREST,
                    technique=technique,
                    predictions=RawPredictions(
                        target=np.array([0, 1]),
                        prediction=np.array([0, 1]),
                    ),
                )

        # Step 2: Simulate CLI auto-resume logic
        datasets = [Dataset.TAIWAN_CREDIT]
        latest_exec_id = predictions_repository.get_latest_execution_id(datasets)

        # Should find the partial execution
        assert latest_exec_id == first_exec_id

        # Check if complete
        params = ExperimentParams(
            execution_id=latest_exec_id,
            datasets=datasets,
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 2,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        is_complete = executor1.is_execution_complete(latest_exec_id, params, config)

        # Should be incomplete (2 out of 8 combinations)
        assert is_complete is False

        # Step 3: Resume execution
        executor2 = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=_create_mock_pipeline_executor(predictions_repository),
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        executor2.execute_experiment(params, config)

        # Should now be complete
        final_completed = predictions_repository.get_completed_combinations(latest_exec_id)
        assert len(final_completed) == 8  # 4 techniques * 2 seeds

        # Verify it's now complete
        is_complete_after = executor2.is_execution_complete(latest_exec_id, params, config)
        assert is_complete_after is True


class DescribeGetCompletedCount:
    """Tests for the get_completed_count method."""

    def it_returns_zero_for_new_execution(
        self,
        experiment_executor: ExperimentExecutor,
    ) -> None:
        count = experiment_executor.get_completed_count("non-existent-id")

        assert count == 0

    def it_returns_correct_count_after_execution(
        self,
        experiment_executor: ExperimentExecutor,
    ) -> None:
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        experiment_executor.execute_experiment(params, config)

        count = experiment_executor.get_completed_count(params.execution_id)
        assert count == 4  # 4 techniques * 1 seed


class DescribeIsExecutionComplete:
    """Tests for the is_execution_complete method."""

    def it_returns_false_for_incomplete_execution(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Test that execution is detected as incomplete when combinations remain."""
        execution_id = "partial-execution"

        # Pre-populate with only 2 out of 8 combinations
        for seed in [1, 2]:
            predictions_repository.save_predictions(
                execution_id=execution_id,
                seed=seed,
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                predictions=RawPredictions(
                    target=np.array([0, 1]),
                    prediction=np.array([0, 1]),
                ),
            )

        params = ExperimentParams(
            execution_id=execution_id,
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 2,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        # Only 2 out of 8 combinations (4 techniques * 2 seeds)
        is_complete = experiment_executor.is_execution_complete(execution_id, params, config)
        assert is_complete is False

    def it_returns_true_for_complete_execution(
        self,
        experiment_executor: ExperimentExecutor,
    ) -> None:
        """Test that execution is detected as complete when all combinations done."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 2,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        # Execute all combinations
        experiment_executor.execute_experiment(params, config)

        # Now check if complete
        is_complete = experiment_executor.is_execution_complete(
            params.execution_id, params, config
        )
        assert is_complete is True

    def it_returns_true_when_completed_exceeds_expected(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Test that execution is complete even when more combinations exist than expected.

        This can happen if parameters change (e.g., fewer seeds in the new run).
        """
        execution_id = "over-complete"

        # Pre-populate with 3 seeds worth of data
        for seed in [1, 2, 3]:
            for technique in Technique:
                # Skip CS_SVM for non-SVM models
                if technique == Technique.CS_SVM:
                    continue
                predictions_repository.save_predictions(
                    execution_id=execution_id,
                    seed=seed,
                    dataset=Dataset.TAIWAN_CREDIT,
                    model_type=ModelType.RANDOM_FOREST,
                    technique=technique,
                    predictions=RawPredictions(
                        target=np.array([0, 1]),
                        prediction=np.array([0, 1]),
                    ),
                )

        params = ExperimentParams(
            execution_id=execution_id,
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 2,  # Now only expecting 2 seeds, but have 3
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        # Should be complete (12 completed >= 8 expected)
        is_complete = experiment_executor.is_execution_complete(execution_id, params, config)
        assert is_complete is True


class DescribeExperimentConfig:
    """Tests for ExperimentConfig merging with defaults."""

    def it_uses_default_config_when_none_provided(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
        experiment_settings: ExperimentSettings,
    ) -> None:
        """Test that default config from settings is used."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )

        # Execute without explicit config
        experiment_executor.execute_experiment(params)

        # Should use default num_seeds from experiment_settings (2)
        completed = predictions_repository.get_completed_combinations(params.execution_id)
        assert len(completed) == 8  # 4 techniques * 2 seeds (default)

    def it_overrides_defaults_with_explicit_config(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Test that explicit config overrides defaults."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,  # Override default of 2
        }

        experiment_executor.execute_experiment(params, config)

        # Should use overridden num_seeds (1)
        completed = predictions_repository.get_completed_combinations(params.execution_id)
        assert len(completed) == 4  # 4 techniques * 1 seed


class DescribeValidModelTechniqueCombinations:
    """Tests for valid model/technique combination filtering."""

    def it_excludes_cs_svm_for_non_svm_models(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Verify CS_SVM is excluded for non-SVM model types."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,  # Exclude SVM
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        experiment_executor.execute_experiment(params, config)

        completed = predictions_repository.get_completed_combinations(params.execution_id)
        techniques = {c.technique for c in completed}

        # CS_SVM should NOT be in the completed techniques (only valid for SVM)
        assert Technique.CS_SVM not in techniques
        # Other 4 techniques should be present
        assert len(techniques) == 4

    def it_includes_all_techniques_for_svm(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Verify all techniques including CS_SVM are used for SVM."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.RANDOM_FOREST,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],  # Only SVM
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        experiment_executor.execute_experiment(params, config)

        completed = predictions_repository.get_completed_combinations(params.execution_id)
        techniques = {c.technique for c in completed}

        # All 5 techniques should be present for SVM
        assert techniques == set(Technique)


class DescribePipelineScheduling:
    """Tests for pipeline scheduling logic."""

    def it_schedules_correct_number_of_pipelines(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        predictions_repository: FakePredictionsRepository,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test that the correct number of pipelines are scheduled."""
        mock_executor = _create_mock_pipeline_executor(predictions_repository)
        scheduled_count: list[int] = []

        original_schedule = mock_executor.schedule

        def tracking_schedule(pipeline: Any, initial_state: Any, context: Any) -> None:
            scheduled_count.append(1)
            original_schedule(pipeline, initial_state, context)

        mock_executor.schedule = tracking_schedule  # type: ignore[method-assign]

        executor = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=mock_executor,
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 3,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        executor.execute_experiment(params, config)

        # 1 dataset * 1 model * 4 techniques * 3 seeds = 12 pipelines
        assert len(scheduled_count) == 12

    def it_creates_pipelines_with_correct_context(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        predictions_repository: FakePredictionsRepository,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test that pipelines are created with correct context."""
        mock_executor = _create_mock_pipeline_executor(predictions_repository)
        scheduled_contexts: list[Any] = []

        original_schedule = mock_executor.schedule

        def tracking_schedule(pipeline: Any, initial_state: Any, context: Any) -> None:
            scheduled_contexts.append(context)
            original_schedule(pipeline, initial_state, context)

        mock_executor.schedule = tracking_schedule  # type: ignore[method-assign]

        executor = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=mock_executor,
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 2,
            "use_gpu": True,
        }

        executor.execute_experiment(params, config)

        # Check that contexts have correct values
        for ctx in scheduled_contexts:
            assert ctx.dataset == Dataset.TAIWAN_CREDIT
            assert ctx.model_type == ModelType.RANDOM_FOREST
            assert ctx.use_gpu is True
            assert ctx.n_jobs == 2  # models_n_jobs from config


class DescribeSkipResumeFlag:
    """Tests for the --skip-resume flag behavior."""

    def it_bypasses_auto_resume_with_flag(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test that --skip-resume starts a new execution even when incomplete ones exist."""
        predictions_repository = FakePredictionsRepository()

        # Pre-populate with an incomplete execution
        first_exec_id = "01943aaa-1111-7000-8000-0123456789ab"
        predictions_repository.save_predictions(
            execution_id=first_exec_id,
            seed=1,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            predictions=RawPredictions(
                target=np.array([0, 1]),
                prediction=np.array([0, 1]),
            ),
        )

        # Verify incomplete execution exists
        latest = predictions_repository.get_latest_execution_id([Dataset.TAIWAN_CREDIT])
        assert latest == first_exec_id

        # Now create executor and run with fresh execution (simulating --skip-resume)
        executor = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=_create_mock_pipeline_executor(predictions_repository),
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        # Create NEW execution with different ID (simulating --skip-resume behavior)
        new_exec_id = "01943bbb-2222-7000-8000-0123456789ab"
        params = ExperimentParams(
            execution_id=new_exec_id,  # Explicitly provide new ID
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
        }

        executor.execute_experiment(params, config)

        # Should have two separate executions now
        first_exec_combinations = predictions_repository.get_completed_combinations(first_exec_id)
        new_exec_combinations = predictions_repository.get_completed_combinations(new_exec_id)

        assert len(first_exec_combinations) == 1  # Original incomplete execution
        assert len(new_exec_combinations) == 4  # New complete execution (4 techniques * 1 seed)

        # Verify they are distinct
        assert first_exec_id != new_exec_id


class DescribeSequentialExecution:
    """Integration tests for sequential execution mode."""

    def it_produces_same_results_in_sequential_mode(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Test that sequential mode produces the same combinations as parallel."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 2,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
            "sequential": True,
        }

        experiment_executor.execute_experiment(params, config)

        completed = predictions_repository.get_completed_combinations(params.execution_id)
        assert len(completed) == 8  # 4 techniques * 2 seeds

        # Check both seeds are present
        seeds = {c.seed for c in completed}
        assert seeds == {1, 2}

        # Check all valid techniques are present
        techniques = {c.technique for c in completed}
        assert Technique.CS_SVM not in techniques  # Only valid for SVM
        assert len(techniques) == 4

    def it_saves_predictions_correctly_in_sequential_mode(
        self,
        experiment_executor: ExperimentExecutor,
        predictions_repository: FakePredictionsRepository,
    ) -> None:
        """Test that predictions are saved for each pipeline in sequential mode."""
        params = ExperimentParams(
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
            "sequential": True,
        }

        experiment_executor.execute_experiment(params, config)

        completed = predictions_repository.get_completed_combinations(params.execution_id)
        for combination in completed:
            preds = predictions_repository.get_predictions(params.execution_id, combination)
            assert preds is not None
            assert len(preds.prediction) > 0
            assert len(preds.target) > 0
            assert len(preds.prediction) == len(preds.target)

    def it_resumes_correctly_in_sequential_mode(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        mock_model_trainer: MagicMock,
        mock_data_splitter: MagicMock,
        mock_training_data_loader: MagicMock,
        mock_classifier_factory: MagicMock,
        experiment_settings: ExperimentSettings,
        resource_settings: ResourceSettings,
    ) -> None:
        """Test that auto-resume works in sequential mode."""
        execution_id = "test-sequential-resumption"
        predictions_repository = FakePredictionsRepository()

        # Pre-populate with some completed combinations
        pre_completed = ExperimentCombination(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=1,
        )
        predictions_repository.save_predictions(
            execution_id=execution_id,
            seed=pre_completed.seed,
            dataset=pre_completed.dataset,
            model_type=pre_completed.model_type,
            technique=pre_completed.technique,
            predictions=RawPredictions(
                target=np.array([0, 1]),
                prediction=np.array([0, 1]),
            ),
        )

        executor = ExperimentExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=_create_mock_pipeline_executor(predictions_repository),
            model_trainer=mock_model_trainer,
            data_splitter=mock_data_splitter,
            training_data_loader=mock_training_data_loader,
            classifier_factory=mock_classifier_factory,
            predictions_repository=predictions_repository,
            experiment_settings=experiment_settings,
            resource_settings=resource_settings,
        )

        params = ExperimentParams(
            execution_id=execution_id,
            datasets=[Dataset.TAIWAN_CREDIT],
            excluded_models=[
                ModelType.SVM,
                ModelType.XGBOOST,
                ModelType.MLP,
            ],
        )
        config: ExperimentConfig = {
            "num_seeds": 1,
            "n_jobs": 1,
            "models_n_jobs": 1,
            "use_gpu": False,
            "sequential": True,
        }

        executor.execute_experiment(params, config)

        # Should have completed the remaining 3 techniques (BASELINE already done)
        completed = predictions_repository.get_completed_combinations(execution_id)
        assert len(completed) == 4  # All 4 techniques completed

        # The pre-completed one should still be there
        assert pre_completed in completed
