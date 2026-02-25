"""Tests for grid_search_trainer service."""

from unittest.mock import MagicMock, patch

import numpy as np

from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.training.trainers import ModelTrainRequest, SplitData
from experiments.services.grid_search_trainer import GridSearchModelTrainer


class DescribeGridSearchModelTrainerInit:
    def it_stores_cost_grids(self) -> None:
        cost_grids = [{"C": 1}]

        trainer = GridSearchModelTrainer(cost_grids=cost_grids)

        assert trainer._cost_grids is cost_grids

    def it_defaults_to_balanced_accuracy(self) -> None:
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
    def it_returns_svm_hyperparameters(self, grid_search_trainer: GridSearchModelTrainer) -> None:
        params = grid_search_trainer.get_params_for_classifier(
            model_type=ModelType.SVM, technique=Technique.BASELINE, cost_grids=[]
        )

        assert len(params) > 0
        assert "clf__loss" in params[0]
        assert "clf__alpha" in params[0]

    def it_returns_random_forest_hyperparameters(
        self, grid_search_trainer: GridSearchModelTrainer
    ) -> None:
        params = grid_search_trainer.get_params_for_classifier(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, cost_grids=[]
        )

        assert len(params) > 0
        assert "clf__n_estimators" in params[0]
        assert "clf__max_depth" in params[0]

    def it_returns_xgboost_hyperparameters(
        self, grid_search_trainer: GridSearchModelTrainer
    ) -> None:
        params = grid_search_trainer.get_params_for_classifier(
            model_type=ModelType.XGBOOST, technique=Technique.BASELINE, cost_grids=[]
        )

        assert len(params) > 0
        assert "clf__learning_rate" in params[0]
        assert "clf__max_depth" in params[0]

    def it_returns_mlp_hyperparameters(self, grid_search_trainer: GridSearchModelTrainer) -> None:
        params = grid_search_trainer.get_params_for_classifier(
            model_type=ModelType.MLP, technique=Technique.BASELINE, cost_grids=[]
        )

        assert len(params) > 0
        assert "clf__hidden_layer_sizes" in params[0]
        assert "clf__activation" in params[0]


class DescribeGetParamsForTechnique:
    def it_adds_class_weight_for_cs_svm(self, grid_search_trainer: GridSearchModelTrainer) -> None:
        cost_matrix = [{0: 1, 1: 10}]

        params = grid_search_trainer.get_params_for_classifier(
            model_type=ModelType.SVM, technique=Technique.CS_SVM, cost_grids=cost_matrix
        )

        assert "clf__class_weight" in params[0]
        assert params[0]["clf__class_weight"] == cost_matrix

    def it_disables_mlp_early_stopping_for_random_under_sampling(
        self, grid_search_trainer: GridSearchModelTrainer
    ) -> None:
        params = grid_search_trainer.get_params_for_classifier(
            model_type=ModelType.MLP,
            technique=Technique.RANDOM_UNDER_SAMPLING,
            cost_grids=[],
        )

        assert "clf__early_stopping" in params[0]
        assert params[0]["clf__early_stopping"] == [False]


class DescribeTrain:
    def it_adjusts_cv_folds_for_small_minority_class(self) -> None:
        trainer = GridSearchModelTrainer(cost_grids=[], cv_folds=10)
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # Only 3 minority samples

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

        with patch.object(trainer, "get_params_for_classifier", return_value=[{}]):
            with patch("experiments.services.grid_search_trainer.GridSearchCV") as mock_grid:
                mock_grid_instance = MagicMock()
                mock_grid_instance.best_estimator_ = mock_classifier
                mock_grid_instance.best_params_ = {}
                mock_grid.return_value = mock_grid_instance

                trainer.train(request)

                cv_arg = mock_grid.call_args.kwargs["cv"]
                assert cv_arg.n_splits == 3


class DescribeGetHyperparameters:
    def it_returns_empty_dict_for_unknown_model_type(
        self, grid_search_trainer: GridSearchModelTrainer
    ) -> None:
        unknown_type = MagicMock()

        result = grid_search_trainer._get_hyperparameters(unknown_type)

        assert result == {}
