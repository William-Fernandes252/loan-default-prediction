from typing import cast

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import pytest

from experiments.core.modeling.estimators import MetaCostClassifier, RobustSVC
from experiments.core.modeling.factories import (
    DefaultEstimatorFactory,
    build_pipeline,
    get_hyperparameters,
    get_model_instance,
    get_params_for_technique,
)
from experiments.core.modeling.types import ModelType, Technique


def test_get_model_instance_returns_expected_and_raises_for_unknown() -> None:
    model = get_model_instance(ModelType.SVM, 42)
    assert isinstance(model, RobustSVC)

    with pytest.raises(ValueError):
        get_model_instance(cast(ModelType, "unknown"), 0)


def test_build_pipeline_adds_sampler_and_meta_cost() -> None:
    pipeline_smote = build_pipeline(ModelType.SVM, Technique.SMOTE, random_state=0)
    assert isinstance(pipeline_smote, ImbPipeline)
    assert isinstance(pipeline_smote.named_steps["sampler"], SMOTE)

    pipeline_meta = build_pipeline(ModelType.SVM, Technique.META_COST, random_state=0)
    assert isinstance(pipeline_meta.named_steps["clf"], MetaCostClassifier)


def test_build_pipeline_uses_correct_sampler_variants() -> None:
    rus_pipe = build_pipeline(ModelType.SVM, Technique.RANDOM_UNDER_SAMPLING, random_state=0)
    assert isinstance(rus_pipe.named_steps["sampler"], RandomUnderSampler)

    smote_tomek_pipe = build_pipeline(ModelType.SVM, Technique.SMOTE_TOMEK, random_state=0)
    assert isinstance(smote_tomek_pipe.named_steps["sampler"], SMOTETomek)


def test_get_params_for_technique_meta_cost_pushes_params() -> None:
    base_params = {"clf__C": [1], "sampler__ratio": [0.5]}
    cost_matrix = [{0: 1, 1: 2}]

    params = get_params_for_technique(
        ModelType.SVM,
        Technique.META_COST,
        base_params,
        cost_matrix,
    )

    assert len(params) == 1
    grid = params[0]
    assert grid["clf__cost_matrix"] == cost_matrix
    assert grid["clf__base_estimator__C"] == [1]
    assert grid["sampler__ratio"] == [0.5]


def test_get_params_for_technique_cs_svm_sets_class_weight() -> None:
    base_params = {"clf__C": [1]}
    cost_matrix = [{0: 1, 1: 2}]

    params = get_params_for_technique(
        ModelType.SVM,
        Technique.CS_SVM,
        base_params,
        cost_matrix,
    )

    assert params[0]["clf__class_weight"] == cost_matrix


def test_default_estimator_factory_wraps_functions() -> None:
    factory = DefaultEstimatorFactory()

    pipeline = factory.create_pipeline(ModelType.SVM, Technique.BASELINE, seed=0)
    assert isinstance(pipeline, ImbPipeline)

    grid = factory.get_param_grid(ModelType.SVM, Technique.BASELINE, cost_grids=[{0: 1, 1: 2}])
    assert isinstance(grid, list)
    assert grid  # not empty


def test_get_hyperparameters_returns_known_keys() -> None:
    params = get_hyperparameters(ModelType.MLP)
    assert "clf__hidden_layer_sizes" in params
