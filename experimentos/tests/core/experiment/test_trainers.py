"""Tests for experiments.core.experiment.trainers module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

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


@pytest.fixture
def mock_estimator_factory() -> MagicMock:
    """Create a mock EstimatorFactory."""
    factory = MagicMock()
    factory.create_pipeline.return_value = MagicMock()
    factory.get_param_grid.return_value = [{"clf__C": [1.0]}]
    return factory


class DescribeGridSearchTrainer:
    """Tests for GridSearchTrainer class."""

    def it_initializes_with_default_parameters(
        self, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify default initialization parameters."""
        trainer = GridSearchTrainer(estimator_factory=mock_estimator_factory)

        assert trainer._factory is mock_estimator_factory
        assert trainer._scoring == "roc_auc"
        assert trainer._n_jobs == 1
        assert trainer._verbose == 0

    def it_accepts_custom_parameters(self, mock_estimator_factory: MagicMock) -> None:
        """Verify custom parameters are stored."""
        trainer = GridSearchTrainer(
            estimator_factory=mock_estimator_factory,
            scoring="f1",
            n_jobs=4,
            verbose=2,
        )

        assert trainer._factory is mock_estimator_factory
        assert trainer._scoring == "f1"
        assert trainer._n_jobs == 4
        assert trainer._verbose == 2


class DescribeGridSearchTrainerTrain:
    """Tests for GridSearchTrainer.train() method."""

    def it_returns_trained_model(
        self, sample_split_data: SplitData, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify train returns a TrainedModel."""
        # Setup mock grid search
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {"C": 1.0}
        mock_grid.fit.return_value = mock_grid

        trainer = GridSearchTrainer(
            estimator_factory=mock_estimator_factory, n_jobs=1, verbose=0
        )

        with patch("experiments.core.experiment.trainers.GridSearchCV", return_value=mock_grid):
            result = trainer.train(
                data=sample_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            assert isinstance(result, TrainedModel)
            assert result.estimator is mock_grid.best_estimator_
            assert result.best_params == {"C": 1.0}

    def it_calls_factory_methods(
        self, sample_split_data: SplitData, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify factory methods are called correctly."""
        # Setup mock grid search
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        trainer = GridSearchTrainer(estimator_factory=mock_estimator_factory)

        with patch("experiments.core.experiment.trainers.GridSearchCV", return_value=mock_grid):
            trainer.train(
                data=sample_split_data,
                model_type=ModelType.SVM,
                technique=Technique.SMOTE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            # Verify factory methods were called
            mock_estimator_factory.create_pipeline.assert_called_once_with(
                ModelType.SVM, Technique.SMOTE, 42
            )
            mock_estimator_factory.get_param_grid.assert_called_once_with(
                ModelType.SVM, Technique.SMOTE, []
            )

    def it_adjusts_cv_folds_for_small_class_counts(
        self, imbalanced_split_data: SplitData, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify CV folds are adjusted when class count is low."""
        # Setup mock grid search
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        trainer = GridSearchTrainer(estimator_factory=mock_estimator_factory)

        with patch(
            "experiments.core.experiment.trainers.GridSearchCV", return_value=mock_grid
        ) as mock_grid_cv:
            trainer.train(
                data=imbalanced_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=10,  # Request 10 folds, but minority has only 5
                cost_grids=[],
            )

            # StratifiedKFold should be created with adjusted n_splits
            call_kwargs = mock_grid_cv.call_args[1]
            cv_obj = call_kwargs.get("cv")
            assert cv_obj is not None
            assert cv_obj.n_splits <= 5  # Should be capped at min class count (5) or lower

    def it_uses_correct_scoring_metric(
        self, sample_split_data: SplitData, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify correct scoring metric is used."""
        # Setup mock grid search
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        trainer = GridSearchTrainer(estimator_factory=mock_estimator_factory, scoring="f1")

        with patch(
            "experiments.core.experiment.trainers.GridSearchCV", return_value=mock_grid
        ) as mock_grid_cv:
            trainer.train(
                data=sample_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            call_kwargs = mock_grid_cv.call_args[1]
            assert call_kwargs.get("scoring") == "f1"

    def it_passes_n_jobs_to_grid_search(
        self, sample_split_data: SplitData, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify n_jobs parameter is passed to GridSearchCV."""
        # Setup mock grid search
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        trainer = GridSearchTrainer(estimator_factory=mock_estimator_factory, n_jobs=4)

        with patch(
            "experiments.core.experiment.trainers.GridSearchCV", return_value=mock_grid
        ) as mock_grid_cv:
            trainer.train(
                data=sample_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            call_kwargs = mock_grid_cv.call_args[1]
            assert call_kwargs.get("n_jobs") == 4

    def it_fits_grid_search_on_training_data(
        self, sample_split_data: SplitData, mock_estimator_factory: MagicMock
    ) -> None:
        """Verify grid search is fitted on training data."""
        # Setup mock grid search
        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        trainer = GridSearchTrainer(estimator_factory=mock_estimator_factory)

        with patch("experiments.core.experiment.trainers.GridSearchCV", return_value=mock_grid):
            trainer.train(
                data=sample_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            mock_grid.fit.assert_called_once()
            call_args = mock_grid.fit.call_args[0]
            np.testing.assert_array_equal(call_args[0], sample_split_data.X_train)
            np.testing.assert_array_equal(call_args[1], sample_split_data.y_train)
