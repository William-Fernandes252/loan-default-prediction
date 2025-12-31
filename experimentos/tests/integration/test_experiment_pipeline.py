"""Integration tests for the experiment pipeline.

These tests verify that components actually work together with real data,
ensuring interfaces are compatible and data flows correctly through the pipeline.
"""

from pathlib import Path

import joblib
import numpy as np
import pytest

from experiments.core.data import Dataset
from experiments.core.experiment.evaluators import ClassificationEvaluator
from experiments.core.experiment.persisters import ParquetExperimentPersister
from experiments.core.experiment.pipeline import ExperimentPipeline
from experiments.core.experiment.protocols import (
    DataPaths,
    ExperimentContext,
    ExperimentIdentity,
    TrainingConfig,
)
from experiments.core.experiment.splitters import StratifiedDataSplitter
from experiments.core.experiment.trainers import GridSearchTrainer
from experiments.core.modeling.factories import DefaultEstimatorFactory
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.storage.local import LocalStorageService


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Create sample data for integration testing.

    Returns balanced, linearly separable data that a simple model can learn.
    """
    np.random.seed(42)

    # Create linearly separable data
    # Class 0: centered around [0, 0]
    X_class0 = np.random.randn(50, 10) * 0.5
    # Class 1: centered around [2, 2]
    X_class1 = np.random.randn(50, 10) * 0.5 + 2.0

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 50 + [1] * 50)

    return X, y


@pytest.fixture
def memory_mapped_data(tmp_path: Path, sample_data: tuple[np.ndarray, np.ndarray]) -> DataPaths:
    """Create memory-mapped data files for testing.

    Returns paths to the memory-mapped files.
    """
    X, y = sample_data

    X_path = tmp_path / "X.joblib"
    y_path = tmp_path / "y.joblib"

    joblib.dump(X, X_path)
    joblib.dump(y, y_path)

    return DataPaths(
        X_path=str(X_path),
        y_path=str(y_path),
    )


@pytest.fixture
def experiment_context(tmp_path: Path, memory_mapped_data: DataPaths) -> ExperimentContext:
    """Create a complete experiment context for testing."""
    identity = ExperimentIdentity(
        dataset=Dataset.TAIWAN_CREDIT,
        model_type=ModelType.RANDOM_FOREST,
        technique=Technique.BASELINE,
        seed=42,
    )

    training_config = TrainingConfig(
        cv_folds=3,  # Use fewer folds for faster testing
        cost_grids=[],
    )

    checkpoint_uri = str(tmp_path / "checkpoint.parquet")

    return ExperimentContext(
        identity=identity,
        data=memory_mapped_data,
        config=training_config,
        checkpoint_uri=checkpoint_uri,
        discard_checkpoints=False,
    )


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


class DescribeExperimentPipeline:
    """Integration tests for ExperimentPipeline with real components."""

    def it_runs_complete_pipeline_with_real_components(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
    ) -> None:
        """Verify the pipeline runs end-to-end with real components and real data.

        This is a "thin slice" integration test that ensures:
        1. Data flows correctly between components
        2. Component interfaces are compatible
        3. The pipeline produces real, meaningful results
        """
        # Create pipeline with real components (not mocks)
        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # Run the pipeline
        result = pipeline.run(experiment_context)

        # Verify we got a real result
        assert result is not None
        assert result.task_id is not None
        assert result.metrics is not None

        # Verify result has the correct structure
        assert isinstance(result.task_id, str)
        assert "taiwan_credit" in result.task_id
        assert "random_forest" in result.task_id

        # Verify all expected metrics are present
        expected_metrics = [
            "accuracy_balanced",
            "g_mean",
            "f1_score",
            "precision",
            "recall",
            "roc_auc",
            "dataset",
            "seed",
            "model",
            "technique",
            "best_params",
        ]
        for metric in expected_metrics:
            assert metric in result.metrics, f"Missing metric: {metric}"

        # Verify metrics have sensible values (not mock values)
        assert 0.0 <= result.metrics["accuracy_balanced"] <= 1.0
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0
        assert 0.0 <= result.metrics["f1_score"] <= 1.0

        # Verify the model actually learned something
        # With linearly separable data, a real model should get good accuracy
        assert result.metrics["accuracy_balanced"] > 0.6, (
            "Model should learn from the linearly separable data"
        )

        # Verify metadata is correct
        assert result.metrics["dataset"] == "taiwan_credit"
        assert result.metrics["model"] == "random_forest"
        assert result.metrics["technique"] == "baseline"
        assert result.metrics["seed"] == 42

    def it_produces_trained_model_that_can_predict(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify the pipeline returns a functional trained model."""
        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        result = pipeline.run(experiment_context)

        # Verify we got a model
        assert result.model is not None

        # Verify the model can make predictions
        X, _ = sample_data
        predictions = result.model.predict(X[:10])

        assert predictions is not None
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions), "Binary classifier should predict 0 or 1"

    def it_handles_imbalanced_data(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
        tmp_path: Path,
    ) -> None:
        """Verify the pipeline handles imbalanced data correctly."""
        # Create imbalanced data
        np.random.seed(42)
        X_class0 = np.random.randn(90, 10) * 0.5
        X_class1 = np.random.randn(10, 10) * 0.5 + 2.0
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 90 + [1] * 10)

        # Save as memory-mapped files
        X_path = tmp_path / "X_imbalanced.joblib"
        y_path = tmp_path / "y_imbalanced.joblib"
        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

        # Update context with imbalanced data
        imbalanced_context = ExperimentContext(
            identity=experiment_context.identity,
            data=DataPaths(X_path=str(X_path), y_path=str(y_path)),
            config=experiment_context.config,
            checkpoint_uri=str(tmp_path / "checkpoint_imbalanced.parquet"),
            discard_checkpoints=False,
        )

        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        result = pipeline.run(imbalanced_context)

        # Pipeline should handle imbalanced data without crashing
        assert result is not None
        assert result.task_id is not None

        # Metrics should still be computed
        assert "g_mean" in result.metrics
        assert "accuracy_balanced" in result.metrics

    def it_respects_checkpoint_skipping(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
    ) -> None:
        """Verify the pipeline skips execution when checkpoint exists."""
        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # First run - should execute
        result1 = pipeline.run(experiment_context)
        assert result1.task_id is not None

        # Second run - should skip (checkpoint exists)
        result2 = pipeline.run(experiment_context)
        assert result2.task_id is None  # Skipped
        assert result2.metrics == {}

    def it_discards_checkpoint_when_requested(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
    ) -> None:
        """Verify the pipeline discards checkpoints when requested."""
        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # First run - creates checkpoint
        result1 = pipeline.run(experiment_context)
        assert result1.task_id is not None

        # Update context to discard checkpoints
        context_with_discard = ExperimentContext(
            identity=experiment_context.identity,
            data=experiment_context.data,
            config=experiment_context.config,
            checkpoint_uri=experiment_context.checkpoint_uri,
            discard_checkpoints=True,
        )

        # Second run - should re-execute
        result2 = pipeline.run(context_with_discard)
        assert result2.task_id is not None  # Executed, not skipped
        assert "roc_auc" in result2.metrics

    def it_works_with_different_model_types(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
    ) -> None:
        """Verify the pipeline works with different model types."""
        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # Test with SVM
        svm_context = ExperimentContext(
            identity=ExperimentIdentity(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.SVM,
                technique=Technique.BASELINE,
                seed=42,
            ),
            data=experiment_context.data,
            config=experiment_context.config,
            checkpoint_uri=experiment_context.checkpoint_uri.replace(
                "checkpoint.parquet", "checkpoint_svm.parquet"
            ),
            discard_checkpoints=False,
        )

        result = pipeline.run(svm_context)

        assert result.task_id is not None
        assert result.metrics["model"] == "svm"
        assert result.metrics["accuracy_balanced"] > 0.5

    def it_works_with_different_techniques(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
    ) -> None:
        """Verify the pipeline works with different imbalance techniques."""
        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # Test with SMOTE
        smote_context = ExperimentContext(
            identity=ExperimentIdentity(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.SMOTE,
                seed=42,
            ),
            data=experiment_context.data,
            config=experiment_context.config,
            checkpoint_uri=experiment_context.checkpoint_uri.replace(
                "checkpoint.parquet", "checkpoint_smote.parquet"
            ),
            discard_checkpoints=False,
        )

        result = pipeline.run(smote_context)

        assert result.task_id is not None
        assert result.metrics["technique"] == "smote"
        assert result.metrics["accuracy_balanced"] > 0.5

    def it_properly_handles_memory_mapped_file_lifecycle(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
        tmp_path: Path,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify memory-mapped files are properly handled and can be cleaned up.

        This test validates that:
        1. Memory-mapped files are created and accessible
        2. The pipeline can read from them
        3. No file handles are left open after pipeline execution
        4. Files can be deleted after the pipeline completes
        """
        X, y = sample_data

        # Create memory-mapped files in a temporary directory
        mmap_dir = tmp_path / "mmap_data"
        mmap_dir.mkdir()

        X_mmap_path = mmap_dir / "X_test.joblib"
        y_mmap_path = mmap_dir / "y_test.joblib"

        # Save data as memory-mapped files
        joblib.dump(X, X_mmap_path)
        joblib.dump(y, y_mmap_path)

        # Verify files were created
        assert X_mmap_path.exists(), "X memory-mapped file should exist"
        assert y_mmap_path.exists(), "y memory-mapped file should exist"

        # Get file sizes for verification
        x_size_before = X_mmap_path.stat().st_size
        y_size_before = y_mmap_path.stat().st_size
        assert x_size_before > 0, "X file should have content"
        assert y_size_before > 0, "y file should have content"

        # Create context with memory-mapped file paths
        mmap_context = ExperimentContext(
            identity=experiment_context.identity,
            data=DataPaths(X_path=str(X_mmap_path), y_path=str(y_mmap_path)),
            config=experiment_context.config,
            checkpoint_uri=str(tmp_path / "checkpoint_mmap.parquet"),
            discard_checkpoints=False,
        )

        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # Run the pipeline
        result = pipeline.run(mmap_context)

        # Verify pipeline completed successfully
        assert result.task_id is not None, "Pipeline should complete successfully"
        assert result.metrics["accuracy_balanced"] > 0.5

        # Verify files still exist after pipeline execution
        assert X_mmap_path.exists(), "X file should still exist after pipeline"
        assert y_mmap_path.exists(), "y file should still exist after pipeline"

        # Verify files can be read after pipeline execution
        # (ensures no corruption or locked file handles)
        X_after = joblib.load(X_mmap_path)
        y_after = joblib.load(y_mmap_path)
        assert X_after.shape == X.shape, "X data should be intact"
        assert y_after.shape == y.shape, "y data should be intact"
        np.testing.assert_array_equal(X_after, X, err_msg="X data should be unchanged")
        np.testing.assert_array_equal(y_after, y, err_msg="y data should be unchanged")

        # Critical test: Verify files can be deleted
        # This ensures no file handles are left open by the pipeline
        try:
            X_mmap_path.unlink()
            y_mmap_path.unlink()
        except PermissionError as e:
            pytest.fail(f"Cannot delete memory-mapped files after pipeline: {e}")

        # Verify deletion succeeded
        assert not X_mmap_path.exists(), "X file should be deleted"
        assert not y_mmap_path.exists(), "y file should be deleted"

    def it_handles_memory_mapped_files_with_multiple_pipeline_runs(
        self,
        experiment_context: ExperimentContext,
        storage: LocalStorageService,
        tmp_path: Path,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify memory-mapped files can be reused across multiple pipeline runs.

        This validates that the pipeline doesn't lock files or leave handles open
        that would prevent subsequent reads.
        """
        X, y = sample_data

        # Create memory-mapped files
        mmap_dir = tmp_path / "mmap_multirun"
        mmap_dir.mkdir()

        X_mmap_path = mmap_dir / "X_multirun.joblib"
        y_mmap_path = mmap_dir / "y_multirun.joblib"

        joblib.dump(X, X_mmap_path)
        joblib.dump(y, y_mmap_path)

        estimator_factory = DefaultEstimatorFactory(use_gpu=False)

        pipeline = ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=0.30),
            trainer=GridSearchTrainer(
                estimator_factory=estimator_factory,
                scoring="roc_auc",
                n_jobs=1,
                verbose=0,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(storage=storage),
        )

        # Run pipeline multiple times with different seeds
        for seed in [42, 43, 44]:
            context = ExperimentContext(
                identity=ExperimentIdentity(
                    dataset=Dataset.TAIWAN_CREDIT,
                    model_type=ModelType.RANDOM_FOREST,
                    technique=Technique.BASELINE,
                    seed=seed,
                ),
                data=DataPaths(X_path=str(X_mmap_path), y_path=str(y_mmap_path)),
                config=TrainingConfig(cv_folds=3, cost_grids=[]),
                checkpoint_uri=str(tmp_path / f"checkpoint_seed{seed}.parquet"),
                discard_checkpoints=False,
            )

            result = pipeline.run(context)
            assert result.task_id is not None, f"Run with seed={seed} should succeed"

        # Verify files can still be accessed and deleted after all runs
        assert X_mmap_path.exists()
        assert y_mmap_path.exists()

        # Should be able to load data
        X_final = joblib.load(X_mmap_path)
        assert X_final.shape == X.shape

        # Should be able to delete files
        X_mmap_path.unlink()
        y_mmap_path.unlink()
        assert not X_mmap_path.exists()
        assert not y_mmap_path.exists()


__all__ = ["DescribeExperimentPipeline"]
