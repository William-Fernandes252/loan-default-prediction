"""Integration tests for training execution.

These tests verify the complete training pipeline integration,
using real implementations of all components but with synthetic data
to ensure fast execution and reproducibility.
"""

import numpy as np
import polars as pl
import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.training.data import TrainingData
from experiments.core.training.trainers import TrainedModel
from experiments.lib.pipelines import PipelineExecutor
from experiments.pipelines.training.factory import TrainingPipelineFactory
from experiments.services.grid_search_trainer import GridSearchModelTrainer
from experiments.services.stratified_data_splitter import StratifiedDataSplitter
from experiments.services.training_executor import TrainingExecutor, TrainingParams
from experiments.services.unbalanced_learner_factory import UnbalancedLearnerFactory

# =============================================================================
# Fixtures for synthetic data generation
# =============================================================================


@pytest.fixture
def synthetic_training_data() -> TrainingData:
    """Generate synthetic training data for testing.

    Creates a balanced binary classification dataset with 200 samples
    and 10 features, suitable for quick integration tests.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create balanced binary target
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    np.random.shuffle(y)

    # Convert to Polars LazyFrames
    X_df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(n_features)})
    y_df = pl.DataFrame({"target": y})

    return TrainingData(X=X_df, y=y_df)


@pytest.fixture
def imbalanced_training_data() -> TrainingData:
    """Generate imbalanced synthetic data (80% class 0, 20% class 1)."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)

    # Imbalanced: 160 class 0, 40 class 1
    y = np.array([0] * 160 + [1] * 40)
    np.random.shuffle(y)

    X_df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(n_features)})
    y_df = pl.DataFrame({"target": y})

    return TrainingData(X=X_df, y=y_df)


class FakeTrainingDataLoader:
    """Fake training data loader that returns pre-configured data."""

    def __init__(self, training_data: TrainingData):
        self._data = training_data

    def load_training_data(self, dataset: Dataset) -> TrainingData:
        return self._data


@pytest.fixture
def training_data_loader(synthetic_training_data: TrainingData) -> FakeTrainingDataLoader:
    """Fixture providing a fake training data loader with synthetic data."""
    return FakeTrainingDataLoader(synthetic_training_data)


@pytest.fixture
def imbalanced_data_loader(imbalanced_training_data: TrainingData) -> FakeTrainingDataLoader:
    """Fixture providing a fake training data loader with imbalanced data."""
    return FakeTrainingDataLoader(imbalanced_training_data)


# =============================================================================
# Fixtures for real service implementations
# =============================================================================


@pytest.fixture
def data_splitter() -> StratifiedDataSplitter:
    """Fixture providing a real stratified data splitter."""
    return StratifiedDataSplitter(test_size=0.3, cv_folds=2)


@pytest.fixture
def classifier_factory() -> UnbalancedLearnerFactory:
    """Fixture providing a real classifier factory (CPU-only)."""
    return UnbalancedLearnerFactory(use_gpu=False)


@pytest.fixture
def model_trainer() -> GridSearchModelTrainer:
    """Fixture providing a real grid search trainer with minimal grid."""
    # Use minimal hyperparameter grids for fast tests
    return GridSearchModelTrainer(
        cost_grids=[{"C": 1.0}],
        scoring="balanced_accuracy",
        cv_folds=2,
        verbose=False,
    )


@pytest.fixture
def pipeline_executor() -> PipelineExecutor:
    """Fixture providing a real pipeline executor."""
    return PipelineExecutor(max_workers=1)


@pytest.fixture
def training_pipeline_factory() -> TrainingPipelineFactory:
    """Fixture providing a real training pipeline factory."""
    return TrainingPipelineFactory()


@pytest.fixture
def seed_generator():
    """Fixture providing a deterministic seed generator."""
    return lambda: 42


@pytest.fixture
def training_executor(
    training_pipeline_factory: TrainingPipelineFactory,
    pipeline_executor: PipelineExecutor,
    model_trainer: GridSearchModelTrainer,
    data_splitter: StratifiedDataSplitter,
    training_data_loader: FakeTrainingDataLoader,
    classifier_factory: UnbalancedLearnerFactory,
    seed_generator,
) -> TrainingExecutor:
    """Fixture providing a fully wired TrainingExecutor."""
    return TrainingExecutor(
        training_pipeline_factory=training_pipeline_factory,
        pipeline_executor=pipeline_executor,
        model_trainer=model_trainer,
        data_splitter=data_splitter,
        training_data_loader=training_data_loader,
        classifier_factory=classifier_factory,
        seed_generator=seed_generator,
    )


# =============================================================================
# Integration tests for TrainingExecutor
# =============================================================================


class DescribeTrainingExecutor:
    """Integration tests for the TrainingExecutor service."""

    class DescribeTrainModel:
        """Tests for the train_model method."""

        def it_trains_random_forest_baseline(self, training_executor: TrainingExecutor) -> None:
            params = TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                n_jobs=1,
                use_gpu=False,
            )

            result = training_executor.train_model(params)

            assert isinstance(result, TrainedModel)
            assert result.model is not None
            assert result.seed == 42

        def it_trains_random_forest_with_smote(self, training_executor: TrainingExecutor) -> None:
            params = TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.SMOTE,
                n_jobs=1,
                use_gpu=False,
            )

            result = training_executor.train_model(params)

            assert isinstance(result, TrainedModel)
            assert result.model is not None

        def it_trains_random_forest_with_random_under_sampling(
            self, training_executor: TrainingExecutor
        ) -> None:
            params = TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.RANDOM_UNDER_SAMPLING,
                n_jobs=1,
                use_gpu=False,
            )

            result = training_executor.train_model(params)

            assert isinstance(result, TrainedModel)
            assert result.model is not None

        def it_trains_random_forest_with_smote_tomek(
            self, training_executor: TrainingExecutor
        ) -> None:
            params = TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.SMOTE_TOMEK,
                n_jobs=1,
                use_gpu=False,
            )

            result = training_executor.train_model(params)

            assert isinstance(result, TrainedModel)
            assert result.model is not None

        def it_trains_random_forest_with_cs_svm_technique(
            self, training_executor: TrainingExecutor
        ) -> None:
            params = TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.CS_SVM,
                n_jobs=1,
                use_gpu=False,
            )

            result = training_executor.train_model(params)

            assert isinstance(result, TrainedModel)
            assert result.model is not None

        def it_trained_model_can_make_predictions(
            self, training_executor: TrainingExecutor, synthetic_training_data: TrainingData
        ) -> None:
            params = TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                n_jobs=1,
                use_gpu=False,
            )

            result = training_executor.train_model(params)

            # Get test data to predict on
            X_test = synthetic_training_data.X.to_numpy()[:10]
            predictions = result.model.predict(X_test)

            assert len(predictions) == 10
            assert all(p in [0, 1] for p in predictions)


class DescribeTrainingWithDifferentModelTypes:
    """Integration tests for training with different model types."""

    @pytest.mark.parametrize(
        "model_type",
        [
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
        ],
    )
    def it_trains_model_type_with_baseline(
        self,
        model_type: ModelType,
        training_executor: TrainingExecutor,
    ) -> None:
        params = TrainingParams(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=model_type,
            technique=Technique.BASELINE,
            n_jobs=1,
            use_gpu=False,
        )

        result = training_executor.train_model(params)

        assert isinstance(result, TrainedModel)
        assert result.model is not None


class DescribeTrainingWithImbalancedData:
    """Integration tests for training with imbalanced datasets."""

    @pytest.fixture
    def imbalanced_training_executor(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        pipeline_executor: PipelineExecutor,
        model_trainer: GridSearchModelTrainer,
        data_splitter: StratifiedDataSplitter,
        imbalanced_data_loader: FakeTrainingDataLoader,
        classifier_factory: UnbalancedLearnerFactory,
        seed_generator,
    ) -> TrainingExecutor:
        """Fixture providing a TrainingExecutor with imbalanced data."""
        return TrainingExecutor(
            training_pipeline_factory=training_pipeline_factory,
            pipeline_executor=pipeline_executor,
            model_trainer=model_trainer,
            data_splitter=data_splitter,
            training_data_loader=imbalanced_data_loader,
            classifier_factory=classifier_factory,
            seed_generator=seed_generator,
        )

    def it_trains_with_smote_on_imbalanced_data(
        self, imbalanced_training_executor: TrainingExecutor
    ) -> None:
        params = TrainingParams(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            n_jobs=1,
            use_gpu=False,
        )

        result = imbalanced_training_executor.train_model(params)

        assert isinstance(result, TrainedModel)
        assert result.model is not None

    def it_trains_with_random_undersampling_on_imbalanced_data(
        self, imbalanced_training_executor: TrainingExecutor
    ) -> None:
        params = TrainingParams(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.RANDOM_UNDER_SAMPLING,
            n_jobs=1,
            use_gpu=False,
        )

        result = imbalanced_training_executor.train_model(params)

        assert isinstance(result, TrainedModel)
        assert result.model is not None


class DescribeTrainingParams:
    """Tests for TrainingParams validation."""

    def it_validates_svm_with_cs_svm_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="Cost-sensitive SVM is not supported for SVM"):
            TrainingParams(
                dataset=Dataset.TAIWAN_CREDIT,
                model_type=ModelType.SVM,
                technique=Technique.CS_SVM,
                n_jobs=1,
                use_gpu=False,
            )

    def it_accepts_valid_model_type_and_technique_combinations(self) -> None:
        # Random Forest with CS_SVM technique is valid
        params = TrainingParams(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.CS_SVM,
            n_jobs=1,
            use_gpu=False,
        )

        assert params.model_type == ModelType.RANDOM_FOREST
        assert params.technique == Technique.CS_SVM


class DescribeTrainingPipelineExecution:
    """Integration tests for the training pipeline execution flow."""

    def it_executes_all_pipeline_steps(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        pipeline_executor: PipelineExecutor,
        data_splitter: StratifiedDataSplitter,
        training_data_loader: FakeTrainingDataLoader,
        classifier_factory: UnbalancedLearnerFactory,
        model_trainer: GridSearchModelTrainer,
    ) -> None:
        """Verify that all pipeline steps execute successfully."""
        from experiments.pipelines.training.pipeline import (
            TrainingPipelineContext,
            TrainingPipelineState,
        )

        pipeline = training_pipeline_factory.create_pipeline(name="TestPipeline")

        context = TrainingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            data_splitter=data_splitter,
            training_data_loader=training_data_loader,
            classifier_factory=classifier_factory,
            trainer=model_trainer,
            seed=42,
            n_jobs=1,
            use_gpu=False,
        )

        result = pipeline_executor.execute(pipeline, TrainingPipelineState(), context)

        assert result.succeeded()
        assert "trained_model" in result.final_state
        assert "predictions" in result.final_state
        assert "data_split" in result.final_state

    def it_pipeline_makes_predictions_after_training(
        self,
        training_pipeline_factory: TrainingPipelineFactory,
        pipeline_executor: PipelineExecutor,
        data_splitter: StratifiedDataSplitter,
        training_data_loader: FakeTrainingDataLoader,
        classifier_factory: UnbalancedLearnerFactory,
        model_trainer: GridSearchModelTrainer,
    ) -> None:
        """Verify that the pipeline produces predictions."""
        from experiments.pipelines.training.pipeline import (
            TrainingPipelineContext,
            TrainingPipelineState,
        )

        pipeline = training_pipeline_factory.create_pipeline(name="TestPipeline")

        context = TrainingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            data_splitter=data_splitter,
            training_data_loader=training_data_loader,
            classifier_factory=classifier_factory,
            trainer=model_trainer,
            seed=42,
            n_jobs=1,
            use_gpu=False,
        )

        result = pipeline_executor.execute(pipeline, TrainingPipelineState(), context)

        predictions = result.final_state["predictions"]
        assert predictions.prediction is not None
        assert predictions.target is not None
        assert len(predictions.prediction) == len(predictions.target)
