"""Tests for SimpleModelTrainer."""

from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.training.trainers import ModelTrainRequest, TrainedModel
from experiments.services.simple_trainer import SimpleModelTrainer


class DescribeSimpleModelTrainerInit:
    """Test SimpleModelTrainer initialization."""

    def it_initializes_without_cost_grids(self) -> None:
        trainer = SimpleModelTrainer()
        assert trainer is not None
        assert trainer._cost_grids is None

    def it_initializes_with_cost_grids(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=["balanced"])
        assert trainer._cost_grids == ["balanced"]


class DescribeGetDefaultParams:
    """Test default parameter generation."""

    def it_returns_svm_defaults(self) -> None:
        trainer = SimpleModelTrainer()
        params = trainer._get_default_params(ModelType.SVM)

        assert params["clf__loss"] == "log_loss"
        assert params["clf__alpha"] == 0.0001
        assert params["clf__penalty"] == "l2"
        assert params["clf__max_iter"] == 1000
        assert params["clf__tol"] == 1e-3

    def it_returns_random_forest_defaults(self) -> None:
        trainer = SimpleModelTrainer()
        params = trainer._get_default_params(ModelType.RANDOM_FOREST)

        assert params["clf__n_estimators"] == 100
        assert params["clf__max_depth"] is None
        assert params["clf__min_samples_leaf"] == 1

    def it_returns_xgboost_defaults(self) -> None:
        trainer = SimpleModelTrainer()
        params = trainer._get_default_params(ModelType.XGBOOST)

        assert params["clf__n_estimators"] == 100
        assert params["clf__learning_rate"] == 0.1
        assert params["clf__max_depth"] == 6
        assert params["clf__subsample"] == 1.0
        assert params["clf__colsample_bytree"] == 1.0
        assert params["clf__reg_alpha"] == 0
        assert params["clf__reg_lambda"] == 1.0

    def it_returns_mlp_defaults(self) -> None:
        trainer = SimpleModelTrainer()
        params = trainer._get_default_params(ModelType.MLP)

        assert params["clf__hidden_layer_sizes"] == (100,)
        assert params["clf__activation"] == "relu"
        assert params["clf__alpha"] == 0.0001
        assert params["clf__early_stopping"] is True


class DescribeApplyTechniqueAdjustments:
    """Test technique-specific parameter adjustments."""

    def it_adds_class_weight_for_cs_svm(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=["balanced"])
        params = {"clf__alpha": 0.0001}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert adjusted["clf__class_weight"] == "balanced"

    def it_uses_first_non_none_cost_grid_for_cs_svm(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=[{0: 1, 1: 5}, "balanced"])
        params = {}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert adjusted["clf__class_weight"] == {0: 1, 1: 5}

    def it_skips_none_entries_in_cost_grids(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=[None, "balanced", {0: 1, 1: 5}])
        params = {}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert adjusted["clf__class_weight"] == "balanced"

    def it_defaults_to_balanced_when_no_cost_grids(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=None)
        params = {}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert adjusted["clf__class_weight"] == "balanced"

    def it_defaults_to_balanced_when_cost_grids_is_empty(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=[])
        params = {}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert adjusted["clf__class_weight"] == "balanced"

    def it_defaults_to_balanced_when_all_cost_grids_are_none(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=[None, None])
        params = {}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert adjusted["clf__class_weight"] == "balanced"

    def it_does_not_add_class_weight_for_cs_svm_with_non_svm_model(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=["balanced"])
        params = {"clf__n_estimators": 100}

        adjusted = trainer._apply_technique_adjustments(
            params, ModelType.RANDOM_FOREST, Technique.CS_SVM
        )

        assert "clf__class_weight" not in adjusted

    def it_disables_early_stopping_for_mlp_with_rus(self) -> None:
        trainer = SimpleModelTrainer()
        params = {"clf__early_stopping": True}

        adjusted = trainer._apply_technique_adjustments(
            params, ModelType.MLP, Technique.RANDOM_UNDER_SAMPLING
        )

        assert adjusted["clf__early_stopping"] is False

    def it_leaves_baseline_params_unchanged(self) -> None:
        trainer = SimpleModelTrainer()
        params = {"clf__alpha": 0.0001}

        adjusted = trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.BASELINE)

        assert adjusted == params

    def it_leaves_smote_params_unchanged(self) -> None:
        trainer = SimpleModelTrainer()
        params = {"clf__n_estimators": 100}

        adjusted = trainer._apply_technique_adjustments(
            params, ModelType.RANDOM_FOREST, Technique.SMOTE
        )

        assert adjusted == params

    def it_leaves_smote_tomek_params_unchanged(self) -> None:
        trainer = SimpleModelTrainer()
        params = {"clf__n_estimators": 100}

        adjusted = trainer._apply_technique_adjustments(
            params, ModelType.RANDOM_FOREST, Technique.SMOTE_TOMEK
        )

        assert adjusted == params

    def it_does_not_mutate_original_params(self) -> None:
        trainer = SimpleModelTrainer(cost_grids=["balanced"])
        params = {"clf__alpha": 0.0001}

        trainer._apply_technique_adjustments(params, ModelType.SVM, Technique.CS_SVM)

        assert "clf__class_weight" not in params


class DescribeTrain:
    """Test model training."""

    def it_trains_model_with_default_params(
        self, balanced_training_data, learner_factory, stratified_splitter
    ) -> None:
        trainer = SimpleModelTrainer()
        split_data = stratified_splitter.split(balanced_training_data, seed=42)

        classifier = learner_factory.create_model(
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            seed=42,
        )

        request = ModelTrainRequest(
            classifier=classifier,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            data=split_data,
            seed=42,
        )

        result = trainer.train(request)

        assert result.model is not None
        assert result.seed == 42
        assert "clf__alpha" in result.params
        assert result.params["clf__alpha"] == 0.0001

    def it_returns_trained_model_structure(
        self, balanced_training_data, learner_factory, stratified_splitter
    ) -> None:
        trainer = SimpleModelTrainer()
        split_data = stratified_splitter.split(balanced_training_data, seed=42)

        classifier = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        request = ModelTrainRequest(
            classifier=classifier,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            data=split_data,
            seed=42,
        )

        result = trainer.train(request)

        # Verify TrainedModel structure
        assert isinstance(result, TrainedModel)
        assert result.model is not None
        assert isinstance(result.params, dict)
        assert result.seed == 42

        # Verify model can make predictions
        predictions = result.model.predict(split_data.X_test)
        assert len(predictions) == len(split_data.y_test)

    def it_applies_cs_svm_technique(
        self, balanced_training_data, learner_factory, stratified_splitter
    ) -> None:
        trainer = SimpleModelTrainer(cost_grids=[{0: 1, 1: 5}])
        split_data = stratified_splitter.split(balanced_training_data, seed=42)

        classifier = learner_factory.create_model(
            model_type=ModelType.SVM,
            technique=Technique.CS_SVM,
            seed=42,
        )

        request = ModelTrainRequest(
            classifier=classifier,
            model_type=ModelType.SVM,
            technique=Technique.CS_SVM,
            data=split_data,
            seed=42,
        )

        result = trainer.train(request)

        assert result.model is not None
        assert "clf__class_weight" in result.params
        assert result.params["clf__class_weight"] == {0: 1, 1: 5}

    def it_applies_mlp_rus_technique(
        self, balanced_training_data, learner_factory, stratified_splitter
    ) -> None:
        trainer = SimpleModelTrainer()
        split_data = stratified_splitter.split(balanced_training_data, seed=42)

        classifier = learner_factory.create_model(
            model_type=ModelType.MLP,
            technique=Technique.RANDOM_UNDER_SAMPLING,
            seed=42,
        )

        request = ModelTrainRequest(
            classifier=classifier,
            model_type=ModelType.MLP,
            technique=Technique.RANDOM_UNDER_SAMPLING,
            data=split_data,
            seed=42,
        )

        result = trainer.train(request)

        assert result.model is not None
        assert "clf__early_stopping" in result.params
        assert result.params["clf__early_stopping"] is False
