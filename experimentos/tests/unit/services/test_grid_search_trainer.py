"""Tests for grid_search_trainer service."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.services.grid_search_trainer import GridSearchModelTrainer


class DescribeGridSearchModelTrainerInit:
    def it_stores_cost_grids(self) -> None:
        cost_grids = [{"C": 1}]
        trainer = GridSearchModelTrainer(cost_grids=cost_grids)

        assert trainer._cost_grids is cost_grids

    def it_defaults_to_balanced_accuracy_scoring(self) -> None:
        trainer = GridSearchModelTrainer(cost_grids=[])

        assert trainer._scoring == "balanced_accuracy"

    def it_accepts_custom_scoring(self) -> None:
        trainer = GridSearchModelTrainer(cost_grids=[], scoring="f1")

        assert trainer._scoring == "f1"

    def it_defaults_to_five_cv_folds(self) -> None:
        trainer = GridSearchModelTrainer(cost_grids=[])

        assert trainer._cv_folds == 5

    def it_accepts_custom_cv_folds(self) -> None:
        trainer = GridSearchModelTrainer(cost_grids=[], cv_folds=10)

        assert trainer._cv_folds == 10

    def it_defaults_to_non_verbose(self) -> None:
        trainer = GridSearchModelTrainer(cost_grids=[])

        assert trainer._verbose is False


class DescribeGetParamsForClassifier:
    @pytest.fixture
    def trainer(self) -> GridSearchModelTrainer:
        return GridSearchModelTrainer(cost_grids=[])

    def it_returns_svm_hyperparameters(self, trainer: GridSearchModelTrainer) -> None:
        params = trainer.get_params_for_classifier(
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            cost_grids=[],
        )

        assert len(params) > 0
        assert "clf__C" in params[0]
        assert "clf__kernel" in params[0]

    def it_returns_random_forest_hyperparameters(self, trainer: GridSearchModelTrainer) -> None:
        params = trainer.get_params_for_classifier(
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            cost_grids=[],
        )

        assert len(params) > 0
        assert "clf__n_estimators" in params[0]
        assert "clf__max_depth" in params[0]

    def it_returns_xgboost_hyperparameters(self, trainer: GridSearchModelTrainer) -> None:
        params = trainer.get_params_for_classifier(
            model_type=ModelType.XGBOOST,
            technique=Technique.BASELINE,
            cost_grids=[],
        )

        assert len(params) > 0
        assert "clf__learning_rate" in params[0]
        assert "clf__max_depth" in params[0]

    def it_returns_mlp_hyperparameters(self, trainer: GridSearchModelTrainer) -> None:
        params = trainer.get_params_for_classifier(
            model_type=ModelType.MLP,
            technique=Technique.BASELINE,
            cost_grids=[],
        )

        assert len(params) > 0
        assert "clf__hidden_layer_sizes" in params[0]
        assert "clf__activation" in params[0]


class DescribeGetParamsForTechnique:
    @pytest.fixture
    def trainer(self) -> GridSearchModelTrainer:
        return GridSearchModelTrainer(cost_grids=[])

    def it_adds_class_weight_for_cs_svm(self, trainer: GridSearchModelTrainer) -> None:
        cost_matrix = [{0: 1, 1: 10}]
        params = trainer.get_params_for_classifier(
            model_type=ModelType.SVM,
            technique=Technique.CS_SVM,
            cost_grids=cost_matrix,
        )

        assert "clf__class_weight" in params[0]
        assert params[0]["clf__class_weight"] == cost_matrix

    def it_wraps_params_for_metacost(self, trainer: GridSearchModelTrainer) -> None:
        cost_matrix = [np.array([[0, 1], [5, 0]])]
        params = trainer.get_params_for_classifier(
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.META_COST,
            cost_grids=cost_matrix,
        )

        # MetaCost should have cost_matrix parameter
        assert "clf__cost_matrix" in params[0]
        # Base estimator params should be prefixed
        assert "clf__base_estimator__n_estimators" in params[0]


class DescribeTrain:
    def it_adjusts_cv_folds_for_small_classes(self) -> None:
        """When minority class has few samples, CV folds should be reduced."""
        from unittest.mock import patch

        from experiments.core.training.trainers import ModelTrainRequest, SplitData

        trainer = GridSearchModelTrainer(cost_grids=[], cv_folds=10)

        # Create data with very small minority class (only 3 samples)
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        mock_classifier = MagicMock()
        mock_classifier.fit = MagicMock(return_value=mock_classifier)

        request = ModelTrainRequest(
            classifier=mock_classifier,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            data=SplitData(
                X_train=X_train,
                X_test=np.random.rand(5, 5),
                y_train=y_train,
                y_test=np.array([0, 0, 1, 1, 1]),
            ),
            seed=42,
        )

        with patch.object(trainer, "get_params_for_classifier") as mock_params:
            mock_params.return_value = [{}]
            with patch("experiments.services.grid_search_trainer.GridSearchCV") as mock_grid:
                mock_grid_instance = MagicMock()
                mock_grid_instance.best_estimator_ = mock_classifier
                mock_grid_instance.best_params_ = {}
                mock_grid.return_value = mock_grid_instance

                trainer.train(request)

                # Verify StratifiedKFold was created with adjusted folds (3, not 10)
                call_kwargs = mock_grid.call_args.kwargs
                cv_arg = call_kwargs["cv"]
                assert cv_arg.n_splits == 3  # min class has 3 samples


class DescribeGetHyperparameters:
    @pytest.fixture
    def trainer(self) -> GridSearchModelTrainer:
        return GridSearchModelTrainer(cost_grids=[])

    def it_returns_empty_dict_for_unknown_model_type(
        self, trainer: GridSearchModelTrainer
    ) -> None:
        # Access private method for testing edge case
        unknown_type = MagicMock()  # Not a valid ModelType

        result = trainer._get_hyperparameters(unknown_type)

        assert result == {}
