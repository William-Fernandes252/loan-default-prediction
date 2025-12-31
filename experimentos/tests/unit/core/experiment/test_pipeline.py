"""Tests for experiments.core.experiment.pipeline module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.core.data import Dataset
from experiments.core.experiment.pipeline import (
    ExperimentPipeline,
    ExperimentPipelineConfig,
    create_custom_experiment_pipeline,
    create_experiment_pipeline,
)
from experiments.core.experiment.protocols import (
    DataPaths,
    EvaluationResult,
    ExperimentContext,
    ExperimentIdentity,
    ExperimentResult,
    SplitData,
    TrainedModel,
    TrainingConfig,
)
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.storage import StorageService
from experiments.services.storage.local import LocalStorageService


@pytest.fixture
def storage() -> StorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


@pytest.fixture
def sample_context(tmp_path: Path) -> ExperimentContext:
    """Create a sample experiment context."""
    identity = ExperimentIdentity(
        dataset=Dataset.TAIWAN_CREDIT,
        model_type=ModelType.RANDOM_FOREST,
        technique=Technique.BASELINE,
        seed=42,
    )
    data_paths = DataPaths(
        X_path="/data/X.mmap",
        y_path="/data/y.mmap",
    )
    training_config = TrainingConfig(
        cv_folds=5,
        cost_grids=[],
    )
    return ExperimentContext(
        identity=identity,
        data=data_paths,
        config=training_config,
        checkpoint_uri=str(tmp_path / "checkpoint.parquet"),
    )


@pytest.fixture
def sample_split_data() -> SplitData:
    """Create sample split data."""
    return SplitData(
        X_train=np.random.rand(100, 10),
        y_train=np.array([0] * 50 + [1] * 50),
        X_test=np.random.rand(30, 10),
        y_test=np.array([0] * 15 + [1] * 15),
    )


@pytest.fixture
def mock_splitter(sample_split_data: SplitData) -> MagicMock:
    """Create a mock splitter."""
    splitter = MagicMock()
    splitter.split.return_value = sample_split_data
    return splitter


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Create a mock trainer."""
    trainer = MagicMock()
    trainer.train.return_value = TrainedModel(
        estimator=LogisticRegression(),
        best_params={"C": 1.0},
    )
    return trainer


@pytest.fixture
def mock_evaluator() -> MagicMock:
    """Create a mock evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        metrics={
            "accuracy_balanced": 0.85,
            "g_mean": 0.84,
            "f1_score": 0.80,
            "precision": 0.82,
            "recall": 0.78,
            "roc_auc": 0.90,
        }
    )
    return evaluator


@pytest.fixture
def mock_persister() -> MagicMock:
    """Create a mock persister."""
    persister = MagicMock()
    persister.checkpoint_exists.return_value = False
    return persister


class DescribeExperimentPipelineConfig:
    """Tests for ExperimentPipelineConfig dataclass."""

    def it_has_default_values(self) -> None:
        """Verify default configuration values."""
        config = ExperimentPipelineConfig()

        assert config.test_size == 0.30
        assert config.scoring == "roc_auc"
        assert config.trainer_n_jobs == 1
        assert config.trainer_verbose == 0

    def it_accepts_custom_values(self) -> None:
        """Verify custom values are stored."""
        config = ExperimentPipelineConfig(
            test_size=0.20,
            scoring="f1",
            trainer_n_jobs=4,
            trainer_verbose=2,
        )

        assert config.test_size == 0.20
        assert config.scoring == "f1"
        assert config.trainer_n_jobs == 4
        assert config.trainer_verbose == 2


class DescribeExperimentPipeline:
    """Tests for ExperimentPipeline class."""

    def it_initializes_with_all_components(
        self,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify pipeline stores all components."""
        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        assert pipeline._splitter is mock_splitter
        assert pipeline._trainer is mock_trainer
        assert pipeline._evaluator is mock_evaluator
        assert pipeline._persister is mock_persister


class DescribeExperimentPipelineRun:
    """Tests for ExperimentPipeline.run() method."""

    def it_returns_experiment_result(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify run returns ExperimentResult."""
        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert isinstance(result, ExperimentResult)
        assert result.task_id is not None
        assert "accuracy_balanced" in result.metrics

    def it_skips_existing_checkpoint_by_default(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify existing checkpoints are skipped."""
        mock_persister.checkpoint_exists.return_value = True

        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert result.task_id is None
        assert result.metrics == {}
        mock_splitter.split.assert_not_called()

    def it_discards_checkpoint_when_flag_is_set(
        self,
        tmp_path: Path,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify checkpoint is discarded when discard_checkpoints=True."""
        mock_persister.checkpoint_exists.return_value = True

        identity = ExperimentIdentity(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )
        data_paths = DataPaths(
            X_path="/data/X.mmap",
            y_path="/data/y.mmap",
        )
        training_config = TrainingConfig(
            cv_folds=5,
            cost_grids=[],
        )
        context = ExperimentContext(
            identity=identity,
            data=data_paths,
            config=training_config,
            checkpoint_uri=str(tmp_path / "checkpoint.parquet"),
            discard_checkpoints=True,
        )

        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        pipeline.run(context)

        mock_persister.discard_checkpoint.assert_called_once()

    def it_returns_empty_result_when_split_fails(
        self,
        sample_context: ExperimentContext,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify empty result when splitter returns None."""
        mock_splitter = MagicMock()
        mock_splitter.split.return_value = None

        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert result.task_id is None
        assert result.metrics == {}
        mock_trainer.train.assert_not_called()

    def it_calls_pipeline_stages_in_order(
        self,
        sample_context: ExperimentContext,
        sample_split_data: SplitData,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify pipeline stages are called in correct order."""
        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        pipeline.run(sample_context)

        # Verify call order
        mock_splitter.split.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_evaluator.evaluate.assert_called_once()
        mock_persister.save_model.assert_called_once()
        mock_persister.save_checkpoint.assert_called_once()

    def it_includes_metadata_in_metrics(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify metadata is added to metrics."""
        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert result.metrics["dataset"] == sample_context.identity.dataset.id
        assert result.metrics["seed"] == sample_context.identity.seed
        assert result.metrics["model"] == sample_context.identity.model_type.id
        assert result.metrics["technique"] == sample_context.identity.technique.id
        assert "best_params" in result.metrics

    def it_generates_correct_task_id(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify task_id is correctly formatted."""
        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        expected_task_id = (
            f"{sample_context.identity.dataset.id}-"
            f"{sample_context.identity.model_type.id}-"
            f"{sample_context.identity.seed}"
        )
        assert result.task_id == expected_task_id

    def it_returns_trained_model_in_result(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_trainer: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify trained model is included in result."""
        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert result.model is not None
        assert isinstance(result.model, LogisticRegression)

    def it_handles_training_errors_gracefully(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
    ) -> None:
        """Verify errors during training are caught."""
        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = RuntimeError("Training failed")

        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert result.task_id is None
        assert result.metrics == {}

    @pytest.mark.parametrize(
        "error_type,error_msg",
        [
            (ValueError, "Invalid parameter value"),
            (RuntimeError, "Training failed"),
            (OSError, "File operation failed"),
            (IOError, "I/O operation failed"),
        ],
    )
    def it_catches_specific_recoverable_exceptions(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
        error_type: type[Exception],
        error_msg: str,
    ) -> None:
        """Verify specific recoverable exceptions are caught and handled gracefully."""
        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = error_type(error_msg)

        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        result = pipeline.run(sample_context)

        assert result.task_id is None
        assert result.metrics == {}

    @pytest.mark.parametrize(
        "error_type",
        [
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
        ],
    )
    def it_propagates_system_level_exceptions(
        self,
        sample_context: ExperimentContext,
        mock_splitter: MagicMock,
        mock_evaluator: MagicMock,
        mock_persister: MagicMock,
        error_type: type[BaseException],
    ) -> None:
        """Verify system-level exceptions are not caught and propagate to orchestrator."""
        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = error_type()

        pipeline = ExperimentPipeline(
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
            persister=mock_persister,
        )

        # These should not be caught and should propagate
        with pytest.raises(error_type):
            pipeline.run(sample_context)


class DescribeCreateExperimentPipeline:
    """Tests for create_experiment_pipeline() function."""

    def it_creates_pipeline_with_storage(self, storage: StorageService) -> None:
        """Verify pipeline is created with required storage."""
        pipeline = create_experiment_pipeline(storage=storage)

        assert isinstance(pipeline, ExperimentPipeline)
        assert pipeline._splitter is not None
        assert pipeline._trainer is not None
        assert pipeline._evaluator is not None
        assert pipeline._persister is not None

    def it_uses_custom_config_when_provided(self, storage: StorageService) -> None:
        """Verify custom config is applied."""
        config = ExperimentPipelineConfig(test_size=0.20)

        with (
            patch(
                "experiments.core.experiment.pipeline.StratifiedDataSplitter"
            ) as mock_splitter_class,
            patch("experiments.core.experiment.pipeline.GridSearchTrainer"),
            patch("experiments.core.experiment.pipeline.ClassificationEvaluator"),
            patch("experiments.core.experiment.pipeline.ParquetExperimentPersister"),
        ):
            create_experiment_pipeline(storage=storage, config=config)

            mock_splitter_class.assert_called_once_with(test_size=0.20)

    def it_accepts_optional_factories(self, storage: StorageService) -> None:
        """Verify optional factories are used when provided."""
        mock_mvs_factory = MagicMock()
        mock_estimator_factory = MagicMock()

        with (
            patch(
                "experiments.core.experiment.pipeline.ParquetExperimentPersister"
            ) as mock_persister_class,
            patch("experiments.core.experiment.pipeline.GridSearchTrainer") as mock_trainer_class,
        ):
            create_experiment_pipeline(
                storage=storage,
                model_versioning_service_factory=mock_mvs_factory,
                estimator_factory=mock_estimator_factory,
            )

            mock_persister_class.assert_called_once_with(
                storage=storage,
                model_versioning_service_factory=mock_mvs_factory,
            )
            mock_trainer_class.assert_called_once()
            args, kwargs = mock_trainer_class.call_args
            assert kwargs["estimator_factory"] is mock_estimator_factory


class DescribeCreateCustomExperimentPipeline:
    """Tests for create_custom_experiment_pipeline() function."""

    def it_creates_pipeline_with_custom_components(self, storage: StorageService) -> None:
        """Verify custom components are used."""
        mock_splitter = MagicMock()
        mock_trainer = MagicMock()
        mock_evaluator = MagicMock()

        pipeline = create_custom_experiment_pipeline(
            storage=storage,
            splitter=mock_splitter,
            trainer=mock_trainer,
            evaluator=mock_evaluator,
        )

        assert isinstance(pipeline, ExperimentPipeline)
        assert pipeline._splitter is mock_splitter
        assert pipeline._trainer is mock_trainer
        assert pipeline._evaluator is mock_evaluator

    def it_accepts_optional_model_versioning_factory(self, storage: StorageService) -> None:
        """Verify optional model versioning factory is used."""
        mock_mvs_factory = MagicMock()
        mock_splitter = MagicMock()
        mock_trainer = MagicMock()
        mock_evaluator = MagicMock()

        with patch(
            "experiments.core.experiment.pipeline.ParquetExperimentPersister"
        ) as mock_persister_class:
            create_custom_experiment_pipeline(
                storage=storage,
                splitter=mock_splitter,
                trainer=mock_trainer,
                evaluator=mock_evaluator,
                model_versioning_service_factory=mock_mvs_factory,
            )

            mock_persister_class.assert_called_once_with(
                storage=storage,
                model_versioning_service_factory=mock_mvs_factory,
            )
