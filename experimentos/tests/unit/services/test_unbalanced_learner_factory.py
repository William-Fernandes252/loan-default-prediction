"""Tests for unbalanced_learner_factory service."""

from unittest.mock import patch

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiments.core.modeling.classifiers import (
    MetaCostClassifier,
    ModelType,
    Technique,
)
from experiments.core.modeling.classifiers import RobustSVC as SVC
from experiments.core.modeling.classifiers import RobustXGBClassifier as XGBClassifier
from experiments.services.unbalanced_learner_factory import UnbalancedLearnerFactory


class DescribeUnbalancedLearnerFactoryInit:
    def it_defaults_to_no_gpu(self) -> None:
        factory = UnbalancedLearnerFactory()

        assert factory._use_gpu is False

    def it_accepts_gpu_flag_when_cuml_available(self) -> None:
        with patch("experiments.services.unbalanced_learner_factory.HAS_CUML", True):
            factory = UnbalancedLearnerFactory(use_gpu=True)

            assert factory._use_gpu is True

    def it_raises_when_gpu_requested_but_cuml_unavailable(self) -> None:
        with patch("experiments.services.unbalanced_learner_factory.HAS_CUML", False):
            with pytest.raises(ImportError) as exc_info:
                UnbalancedLearnerFactory(use_gpu=True)

            assert "cuML is not available" in str(exc_info.value)


class DescribeCreateModelWithRandomForest:
    def it_creates_random_forest_pipeline(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, seed=42
        )

        assert isinstance(result, ImbPipeline)
        assert isinstance(result.named_steps["clf"], RandomForestClassifier)  # type: ignore[union-attr]

    def it_uses_provided_seed(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, seed=123
        )

        clf = result.named_steps["clf"]  # type: ignore[union-attr]
        assert clf.random_state == 123


class DescribeCreateModelWithSVM:
    def it_creates_svm_pipeline(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.SVM, technique=Technique.BASELINE, seed=42
        )

        assert isinstance(result, ImbPipeline)
        assert isinstance(result.named_steps["clf"], SVC)  # type: ignore[union-attr]


class DescribeCreateModelWithXGBoost:
    def it_creates_xgboost_pipeline(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.XGBOOST, technique=Technique.BASELINE, seed=42
        )

        assert isinstance(result, ImbPipeline)
        assert isinstance(result.named_steps["clf"], XGBClassifier)  # type: ignore[union-attr]

    def it_defaults_to_cpu_device(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.XGBOOST, technique=Technique.BASELINE, seed=42
        )

        clf = result.named_steps["clf"]  # type: ignore[union-attr]
        assert clf.device == "cpu"


class DescribeCreateModelWithMLP:
    def it_creates_mlp_pipeline(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.MLP, technique=Technique.BASELINE, seed=42
        )

        assert isinstance(result, ImbPipeline)
        assert isinstance(result.named_steps["clf"], MLPClassifier)  # type: ignore[union-attr]


class DescribeCreateModelWithSamplingTechniques:
    def it_adds_random_under_sampler(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.RANDOM_UNDER_SAMPLING,
            seed=42,
        )

        assert "sampler" in result.named_steps  # type: ignore[operator]
        assert isinstance(result.named_steps["sampler"], RandomUnderSampler)  # type: ignore[union-attr]

    def it_adds_smote_sampler(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.SMOTE, seed=42
        )

        assert "sampler" in result.named_steps  # type: ignore[operator]
        assert isinstance(result.named_steps["sampler"], SMOTE)  # type: ignore[union-attr]

    def it_adds_smote_tomek_sampler(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.SMOTE_TOMEK, seed=42
        )

        assert "sampler" in result.named_steps  # type: ignore[operator]
        assert isinstance(result.named_steps["sampler"], SMOTETomek)  # type: ignore[union-attr]

    def it_wraps_with_metacost_classifier(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.META_COST, seed=42
        )

        assert isinstance(result.named_steps["clf"], MetaCostClassifier)  # type: ignore[union-attr]

    def it_omits_sampler_for_baseline(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, seed=42
        )

        assert "sampler" not in result.named_steps  # type: ignore[operator]


class DescribeCreateModelPipelineStructure:
    def it_starts_with_imputer(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, seed=42
        )

        assert result.steps[0][0] == "imputer"  # type: ignore[union-attr]
        assert isinstance(result.named_steps["imputer"], SimpleImputer)  # type: ignore[union-attr]

    def it_has_scaler_as_second_step(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, seed=42
        )

        assert result.steps[1][0] == "scaler"  # type: ignore[union-attr]
        assert isinstance(result.named_steps["scaler"], StandardScaler)  # type: ignore[union-attr]

    def it_ends_with_classifier(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST, technique=Technique.BASELINE, seed=42
        )

        assert result.steps[-1][0] == "clf"  # type: ignore[union-attr]


class DescribeCreateModelWithNJobs:
    def it_passes_n_jobs_to_random_forest(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
            n_jobs=4,
        )

        clf = result.named_steps["clf"]  # type: ignore[union-attr]
        assert clf.n_jobs == 4

    def it_passes_n_jobs_to_xgboost(self, learner_factory: UnbalancedLearnerFactory) -> None:
        result = learner_factory.create_model(
            model_type=ModelType.XGBOOST,
            technique=Technique.BASELINE,
            seed=42,
            n_jobs=2,
        )

        clf = result.named_steps["clf"]  # type: ignore[union-attr]
        assert clf.n_jobs == 2
