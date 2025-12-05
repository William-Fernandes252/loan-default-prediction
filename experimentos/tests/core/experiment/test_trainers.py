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


class DescribeGridSearchTrainer:
    """Tests for GridSearchTrainer class."""

    def it_initializes_with_default_parameters(self) -> None:
        """Verify default initialization parameters."""
        trainer = GridSearchTrainer()

        assert trainer._scoring == "roc_auc"
        assert trainer._n_jobs == 1
        assert trainer._verbose == 0

    def it_accepts_custom_parameters(self) -> None:
        """Verify custom parameters are stored."""
        trainer = GridSearchTrainer(
            scoring="f1",
            n_jobs=4,
            verbose=2,
        )

        assert trainer._scoring == "f1"
        assert trainer._n_jobs == 4
        assert trainer._verbose == 2


class DescribeGridSearchTrainerTrain:
    """Tests for GridSearchTrainer.train() method."""

    def it_returns_trained_model(self, sample_split_data: SplitData) -> None:
        """Verify train returns a TrainedModel."""
        trainer = GridSearchTrainer(n_jobs=1, verbose=0)

        # Use a mock pipeline that returns a simple estimator
        mock_pipeline = MagicMock()
        mock_pipeline.fit.return_value = mock_pipeline

        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {"C": 1.0}
        mock_grid.fit.return_value = mock_grid

        with (
            patch(
                "experiments.core.experiment.trainers.build_pipeline",
                return_value=mock_pipeline,
            ),
            patch(
                "experiments.core.experiment.trainers.get_hyperparameters",
                return_value={"clf__C": [1.0]},
            ),
            patch(
                "experiments.core.experiment.trainers.get_params_for_technique",
                return_value={"clf__C": [1.0]},
            ),
            patch(
                "experiments.core.experiment.trainers.GridSearchCV",
                return_value=mock_grid,
            ),
        ):
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

    def it_builds_correct_pipeline(self, sample_split_data: SplitData) -> None:
        """Verify correct pipeline is built for model type and technique."""
        trainer = GridSearchTrainer()

        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        with (
            patch("experiments.core.experiment.trainers.build_pipeline") as mock_build,
            patch(
                "experiments.core.experiment.trainers.get_hyperparameters",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.get_params_for_technique",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.GridSearchCV",
                return_value=mock_grid,
            ),
        ):
            mock_build.return_value = MagicMock()

            trainer.train(
                data=sample_split_data,
                model_type=ModelType.SVM,
                technique=Technique.SMOTE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            mock_build.assert_called_once_with(ModelType.SVM, Technique.SMOTE, 42)

    def it_adjusts_cv_folds_for_small_class_counts(self, imbalanced_split_data: SplitData) -> None:
        """Verify CV folds are adjusted when class count is low."""
        trainer = GridSearchTrainer()

        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        with (
            patch(
                "experiments.core.experiment.trainers.build_pipeline",
                return_value=MagicMock(),
            ),
            patch(
                "experiments.core.experiment.trainers.get_hyperparameters",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.get_params_for_technique",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.GridSearchCV",
                return_value=mock_grid,
            ),
            patch("experiments.core.experiment.trainers.StratifiedKFold") as mock_kfold,
        ):
            trainer.train(
                data=imbalanced_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=10,  # Request 10 folds, but minority has only 5
                cost_grids=[],
            )

            # StratifiedKFold should be called with adjusted n_splits
            mock_kfold.assert_called_once()
            call_kwargs = mock_kfold.call_args[1]
            # Should be capped at min class count (5) or lower
            assert call_kwargs["n_splits"] <= 5

    def it_uses_correct_scoring_metric(self, sample_split_data: SplitData) -> None:
        """Verify correct scoring metric is used."""
        trainer = GridSearchTrainer(scoring="f1")

        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        with (
            patch(
                "experiments.core.experiment.trainers.build_pipeline",
                return_value=MagicMock(),
            ),
            patch(
                "experiments.core.experiment.trainers.get_hyperparameters",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.get_params_for_technique",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.GridSearchCV",
                return_value=mock_grid,
            ) as mock_grid_cv,
            patch("experiments.core.experiment.trainers.StratifiedKFold"),
        ):
            trainer.train(
                data=sample_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            mock_grid_cv.assert_called_once()
            call_kwargs = mock_grid_cv.call_args[1]
            assert call_kwargs["scoring"] == "f1"

    def it_passes_n_jobs_to_grid_search(self, sample_split_data: SplitData) -> None:
        """Verify n_jobs parameter is passed to GridSearchCV."""
        trainer = GridSearchTrainer(n_jobs=4)

        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        with (
            patch(
                "experiments.core.experiment.trainers.build_pipeline",
                return_value=MagicMock(),
            ),
            patch(
                "experiments.core.experiment.trainers.get_hyperparameters",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.get_params_for_technique",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.GridSearchCV",
                return_value=mock_grid,
            ) as mock_grid_cv,
            patch("experiments.core.experiment.trainers.StratifiedKFold"),
        ):
            trainer.train(
                data=sample_split_data,
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
                seed=42,
                cv_folds=5,
                cost_grids=[],
            )

            mock_grid_cv.assert_called_once()
            call_kwargs = mock_grid_cv.call_args[1]
            assert call_kwargs["n_jobs"] == 4

    def it_fits_grid_search_on_training_data(self, sample_split_data: SplitData) -> None:
        """Verify grid search is fitted on training data."""
        trainer = GridSearchTrainer()

        mock_grid = MagicMock()
        mock_grid.best_estimator_ = LogisticRegression()
        mock_grid.best_params_ = {}
        mock_grid.fit.return_value = mock_grid

        with (
            patch(
                "experiments.core.experiment.trainers.build_pipeline",
                return_value=MagicMock(),
            ),
            patch(
                "experiments.core.experiment.trainers.get_hyperparameters",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.get_params_for_technique",
                return_value={},
            ),
            patch(
                "experiments.core.experiment.trainers.GridSearchCV",
                return_value=mock_grid,
            ),
            patch("experiments.core.experiment.trainers.StratifiedKFold"),
        ):
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
