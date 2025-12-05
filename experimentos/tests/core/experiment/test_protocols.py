"""Tests for experiments.core.experiment.protocols module."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.core.data import Dataset
from experiments.core.experiment.protocols import (
    DataPaths,
    DataSplitter,
    EvaluationResult,
    ExperimentContext,
    ExperimentIdentity,
    ExperimentPersister,
    ExperimentResult,
    ModelEvaluator,
    ModelTrainer,
    SplitData,
    TrainedModel,
    TrainingConfig,
)
from experiments.core.modeling.types import ModelType, Technique


class DescribeDataPaths:
    """Tests for DataPaths dataclass."""

    def it_stores_data_paths(self) -> None:
        """Verify DataPaths stores X and y paths."""
        data_paths = DataPaths(
            X_path="/data/X.mmap",
            y_path="/data/y.mmap",
        )

        assert data_paths.X_path == "/data/X.mmap"
        assert data_paths.y_path == "/data/y.mmap"

    def it_is_frozen(self) -> None:
        """Verify DataPaths is immutable."""
        data_paths = DataPaths(
            X_path="/data/X.mmap",
            y_path="/data/y.mmap",
        )

        with pytest.raises(AttributeError):
            data_paths.X_path = "/new/path"  # type: ignore[misc]


class DescribeExperimentIdentity:
    """Tests for ExperimentIdentity dataclass."""

    def it_stores_experiment_identity(self) -> None:
        """Verify ExperimentIdentity stores dataset, model, technique, and seed."""
        identity = ExperimentIdentity(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        assert identity.dataset == Dataset.TAIWAN_CREDIT
        assert identity.model_type == ModelType.RANDOM_FOREST
        assert identity.technique == Technique.BASELINE
        assert identity.seed == 42

    def it_is_frozen(self) -> None:
        """Verify ExperimentIdentity is immutable."""
        identity = ExperimentIdentity(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        with pytest.raises(AttributeError):
            identity.seed = 100  # type: ignore[misc]


class DescribeTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def it_stores_training_configuration(self) -> None:
        """Verify TrainingConfig stores cv_folds and cost_grids."""
        config = TrainingConfig(
            cv_folds=5,
            cost_grids=[{"cost": [1, 10]}],
        )

        assert config.cv_folds == 5
        assert config.cost_grids == [{"cost": [1, 10]}]

    def it_is_frozen(self) -> None:
        """Verify TrainingConfig is immutable."""
        config = TrainingConfig(
            cv_folds=5,
            cost_grids=[],
        )

        with pytest.raises(AttributeError):
            config.cv_folds = 10  # type: ignore[misc]


class DescribeExperimentContext:
    """Tests for ExperimentContext dataclass."""

    def it_stores_all_experiment_configuration(self) -> None:
        """Verify ExperimentContext stores all configuration fields."""
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
            cost_grids=[{"cost": [1, 10]}],
        )

        context = ExperimentContext(
            identity=identity,
            data=data_paths,
            config=training_config,
            checkpoint_path=Path("/checkpoints/test.parquet"),
        )

        assert context.identity == identity
        assert context.data == data_paths
        assert context.config == training_config
        assert context.checkpoint_path == Path("/checkpoints/test.parquet")

        # Verify nested access works
        assert context.identity.dataset == Dataset.TAIWAN_CREDIT
        assert context.identity.model_type == ModelType.RANDOM_FOREST
        assert context.identity.technique == Technique.BASELINE
        assert context.identity.seed == 42
        assert context.data.X_path == "/data/X.mmap"
        assert context.data.y_path == "/data/y.mmap"
        assert context.config.cv_folds == 5
        assert context.config.cost_grids == [{"cost": [1, 10]}]

    def it_has_default_discard_checkpoints_false(self) -> None:
        """Verify discard_checkpoints defaults to False."""
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
            checkpoint_path=Path("/checkpoints/test.parquet"),
        )

        assert context.discard_checkpoints is False

    def it_allows_discard_checkpoints_true(self) -> None:
        """Verify discard_checkpoints can be set to True."""
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
            checkpoint_path=Path("/checkpoints/test.parquet"),
            discard_checkpoints=True,
        )

        assert context.discard_checkpoints is True

    def it_is_frozen(self) -> None:
        """Verify ExperimentContext is immutable."""
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
            checkpoint_path=Path("/checkpoints/test.parquet"),
        )

        with pytest.raises(AttributeError):
            context.identity = identity  # type: ignore[misc]


class DescribeSplitData:
    """Tests for SplitData dataclass."""

    def it_stores_train_and_test_arrays(self) -> None:
        """Verify SplitData stores all data arrays."""
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        X_test = np.array([[5, 6]])
        y_test = np.array([1])

        split = SplitData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        np.testing.assert_array_equal(split.X_train, X_train)
        np.testing.assert_array_equal(split.y_train, y_train)
        np.testing.assert_array_equal(split.X_test, X_test)
        np.testing.assert_array_equal(split.y_test, y_test)

    def it_is_mutable(self) -> None:
        """Verify SplitData allows mutation (for memory cleanup)."""
        split = SplitData(
            X_train=np.array([[1, 2]]),
            y_train=np.array([0]),
            X_test=np.array([[3, 4]]),
            y_test=np.array([1]),
        )

        # Should be mutable for memory management purposes
        split.X_train = np.array([[5, 6]])
        np.testing.assert_array_equal(split.X_train, np.array([[5, 6]]))


class DescribeTrainedModel:
    """Tests for TrainedModel dataclass."""

    def it_stores_estimator_and_best_params(self) -> None:
        """Verify TrainedModel stores estimator and parameters."""
        estimator = LogisticRegression()
        best_params = {"C": 1.0, "penalty": "l2"}

        trained = TrainedModel(estimator=estimator, best_params=best_params)

        assert trained.estimator is estimator
        assert trained.best_params == best_params


class DescribeEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def it_stores_metrics_dictionary(self) -> None:
        """Verify EvaluationResult stores metrics."""
        metrics = {
            "accuracy_balanced": 0.85,
            "f1_score": 0.80,
            "roc_auc": 0.90,
        }

        result = EvaluationResult(metrics=metrics)  # type: ignore[arg-type]

        assert result.metrics == metrics
        assert result.metrics["accuracy_balanced"] == 0.85


class DescribeExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def it_stores_task_id_and_metrics(self) -> None:
        """Verify ExperimentResult stores basic fields."""
        metrics = {"accuracy_balanced": 0.85}

        result = ExperimentResult(
            task_id="taiwan_credit-rf-42",
            metrics=metrics,
        )

        assert result.task_id == "taiwan_credit-rf-42"
        assert result.metrics == metrics

    def it_has_optional_model_field(self) -> None:
        """Verify model field defaults to None."""
        result = ExperimentResult(
            task_id="test",
            metrics={},
        )

        assert result.model is None

    def it_stores_model_when_provided(self) -> None:
        """Verify model can be stored."""
        model = LogisticRegression()

        result = ExperimentResult(
            task_id="test",
            metrics={},
            model=model,
        )

        assert result.model is model

    def it_allows_none_task_id(self) -> None:
        """Verify task_id can be None (for skipped experiments)."""
        result = ExperimentResult(
            task_id=None,
            metrics={},
        )

        assert result.task_id is None


class DescribeDataSplitterProtocol:
    """Tests for DataSplitter protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify DataSplitter can be checked at runtime."""

        class ValidSplitter:
            def split(
                self,
                X_mmap_path: str,
                y_mmap_path: str,
                seed: int,
                cv_folds: int,
            ) -> SplitData | None:
                return None

        splitter = ValidSplitter()
        assert isinstance(splitter, DataSplitter)

    def it_rejects_non_conforming_classes(self) -> None:
        """Verify non-conforming classes are rejected."""

        class InvalidSplitter:
            def wrong_method(self) -> None:
                pass

        splitter = InvalidSplitter()
        assert not isinstance(splitter, DataSplitter)


class DescribeModelTrainerProtocol:
    """Tests for ModelTrainer protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify ModelTrainer can be checked at runtime."""

        class ValidTrainer:
            def train(
                self,
                data: SplitData,
                model_type: ModelType,
                technique: Technique,
                seed: int,
                cv_folds: int,
                cost_grids: list,
            ) -> TrainedModel:
                return TrainedModel(
                    estimator=LogisticRegression(),
                    best_params={},
                )

        trainer = ValidTrainer()
        assert isinstance(trainer, ModelTrainer)


class DescribeModelEvaluatorProtocol:
    """Tests for ModelEvaluator protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify ModelEvaluator can be checked at runtime."""

        class ValidEvaluator:
            def evaluate(
                self,
                model: object,
                X_test: np.ndarray,
                y_test: np.ndarray,
            ) -> EvaluationResult:
                return EvaluationResult(metrics={})

        evaluator = ValidEvaluator()
        assert isinstance(evaluator, ModelEvaluator)


class DescribeExperimentPersisterProtocol:
    """Tests for ExperimentPersister protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify ExperimentPersister can be checked at runtime."""

        class ValidPersister:
            def save_checkpoint(
                self,
                metrics: dict,
                checkpoint_path: Path,
            ) -> None:
                pass

            def save_model(
                self,
                model: object,
                context: ExperimentContext,
            ) -> None:
                pass

            def checkpoint_exists(self, checkpoint_path: Path) -> bool:
                return False

            def discard_checkpoint(self, checkpoint_path: Path) -> None:
                pass

        persister = ValidPersister()
        assert isinstance(persister, ExperimentPersister)
