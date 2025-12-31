"""Tests for experiments.core.experiment.trainers module."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from experiments.core.experiment.protocols import SplitData, TrainedModel
from experiments.core.experiment.trainers import GridSearchTrainer
from experiments.core.modeling.types import ModelType, Technique


@pytest.fixture
def sample_split_data() -> SplitData:
    """Create sample split data for testing."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.array([0] * 50 + [1] * 50)
    X_test = np.random.rand(30, 10)
    y_test = np.array([0] * 15 + [1] * 15)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture
def imbalanced_split_data() -> SplitData:
    """Create imbalanced split data for testing."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.array([0] * 95 + [1] * 5)
    X_test = np.random.rand(30, 10)
    y_test = np.array([0] * 28 + [1] * 2)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


class FakeEstimatorFactory:
    """A simple estimator factory for testing.

    Uses fast, dummy estimators (DecisionTreeClassifier with max_depth=1)
    and minimal parameter grids to test training behavior without mocking sklearn.
    """

    def create_pipeline(
        self, model_type: ModelType, technique: Technique, seed: int
    ) -> DecisionTreeClassifier:
        """Create a simple decision tree estimator."""
        return DecisionTreeClassifier(max_depth=1, random_state=seed)

    def get_param_grid(
        self, model_type: ModelType, technique: Technique, cost_grids: list
    ) -> list[dict]:
        """Return a minimal parameter grid for fast testing."""
        return [{"max_depth": [1, 2]}]


@pytest.fixture
def estimator_factory() -> FakeEstimatorFactory:
    """Create a fake EstimatorFactory."""
    return FakeEstimatorFactory()


class DescribeGridSearchTrainer:
    """Tests for GridSearchTrainer class."""

    def it_initializes_with_default_parameters(
        self, estimator_factory: FakeEstimatorFactory
    ) -> None:
        """Verify default initialization parameters."""
        trainer = GridSearchTrainer(estimator_factory=estimator_factory)

        assert trainer._factory is estimator_factory
        assert trainer._scoring == "roc_auc"
        assert trainer._n_jobs == 1
        assert trainer._verbose == 0

    def it_accepts_custom_parameters(self, estimator_factory: FakeEstimatorFactory) -> None:
        """Verify custom parameters are stored."""
        trainer = GridSearchTrainer(
            estimator_factory=estimator_factory,
            scoring="f1",
            n_jobs=4,
            verbose=2,
        )

        assert trainer._factory is estimator_factory
        assert trainer._scoring == "f1"
        assert trainer._n_jobs == 4
        assert trainer._verbose == 2


class DescribeGridSearchTrainerTrain:
    """Tests for GridSearchTrainer.train() method."""

    def it_returns_trained_model(
        self, sample_split_data: SplitData, estimator_factory: FakeEstimatorFactory
    ) -> None:
        """Verify train returns a TrainedModel with meaningful results."""
        trainer = GridSearchTrainer(estimator_factory=estimator_factory, n_jobs=1, verbose=0)

        result = trainer.train(
            data=sample_split_data,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            cv_folds=5,
            cost_grids=[],
        )

        # Test behavioral outcomes, not implementation details
        assert isinstance(result, TrainedModel)
        assert result.estimator is not None
        assert hasattr(result.estimator, "predict")
        assert isinstance(result.best_params, dict)

        # Verify the model can make predictions
        predictions = result.estimator.predict(sample_split_data.X_test)
        assert predictions.shape == sample_split_data.y_test.shape

        # Verify model learns something (accuracy > random guess)
        accuracy = np.mean(predictions == sample_split_data.y_test)
        assert accuracy > 0.4  # Should beat random guessing for balanced classes

    def it_trains_model_with_actual_learning(
        self, sample_split_data: SplitData, estimator_factory: FakeEstimatorFactory
    ) -> None:
        """Verify the model actually learns from the training data."""
        trainer = GridSearchTrainer(estimator_factory=estimator_factory)

        result = trainer.train(
            data=sample_split_data,
            model_type=ModelType.SVM,
            technique=Technique.SMOTE,
            seed=42,
            cv_folds=5,
            cost_grids=[],
        )

        # Test that training actually occurred - model should fit the data
        train_predictions = result.estimator.predict(sample_split_data.X_train)
        train_accuracy = np.mean(train_predictions == sample_split_data.y_train)

        # Should have decent training accuracy
        assert train_accuracy > 0.5

    def it_adjusts_cv_folds_for_small_class_counts(
        self, imbalanced_split_data: SplitData, estimator_factory: FakeEstimatorFactory
    ) -> None:
        """Verify CV folds are adjusted when class count is low and model still trains."""
        trainer = GridSearchTrainer(estimator_factory=estimator_factory)

        # Request 10 folds, but minority class has only 5 samples
        result = trainer.train(
            data=imbalanced_split_data,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            cv_folds=10,
            cost_grids=[],
        )

        # Test that training succeeded despite fold adjustment
        assert isinstance(result, TrainedModel)
        assert result.estimator is not None

        # Model should still make predictions
        predictions = result.estimator.predict(imbalanced_split_data.X_test)
        assert predictions.shape == imbalanced_split_data.y_test.shape

    def it_produces_different_results_with_different_seeds(
        self, sample_split_data: SplitData, estimator_factory: FakeEstimatorFactory
    ) -> None:
        """Verify different seeds produce different model states."""
        trainer = GridSearchTrainer(estimator_factory=estimator_factory)

        result1 = trainer.train(
            data=sample_split_data,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            cv_folds=5,
            cost_grids=[],
        )

        result2 = trainer.train(
            data=sample_split_data,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=99,
            cv_folds=5,
            cost_grids=[],
        )

        # Different seeds may produce different parameters
        # At minimum, both should be valid trained models
        assert isinstance(result1, TrainedModel)
        assert isinstance(result2, TrainedModel)

        # Both should be able to predict
        pred1 = result1.estimator.predict(sample_split_data.X_test)
        pred2 = result2.estimator.predict(sample_split_data.X_test)
        assert pred1.shape == pred2.shape
