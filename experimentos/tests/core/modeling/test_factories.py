"""Tests for DefaultEstimatorFactory."""

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import pytest
from sklearn.ensemble import RandomForestClassifier

from experiments.core.modeling.estimators import MetaCostClassifier, RobustSVC, RobustXGBClassifier
from experiments.core.modeling.factories import DefaultEstimatorFactory
from experiments.core.modeling.types import ModelType, Technique


@pytest.fixture
def factory() -> DefaultEstimatorFactory:
    """Create a factory instance for testing."""
    return DefaultEstimatorFactory(use_gpu=False)


class DescribeDefaultEstimatorFactory:
    """Tests for DefaultEstimatorFactory class."""

    class DescribeCreatePipeline:
        """Tests for create_pipeline method."""

        def it_creates_baseline_pipeline_with_svm(self, factory: DefaultEstimatorFactory) -> None:
            """Verify baseline pipeline contains expected steps."""
            pipeline = factory.create_pipeline(ModelType.SVM, Technique.BASELINE, seed=42)

            assert isinstance(pipeline, ImbPipeline)
            assert "imputer" in pipeline.named_steps
            assert "scaler" in pipeline.named_steps
            assert "clf" in pipeline.named_steps
            assert isinstance(pipeline.named_steps["clf"], RobustSVC)
            assert "sampler" not in pipeline.named_steps

        def it_creates_pipeline_with_smote_sampler(self, factory: DefaultEstimatorFactory) -> None:
            """Verify SMOTE technique adds sampler step."""
            pipeline = factory.create_pipeline(ModelType.SVM, Technique.SMOTE, seed=0)

            assert isinstance(pipeline, ImbPipeline)
            assert isinstance(pipeline.named_steps["sampler"], SMOTE)

        def it_creates_pipeline_with_random_under_sampling(
            self, factory: DefaultEstimatorFactory
        ) -> None:
            """Verify RANDOM_UNDER_SAMPLING adds correct sampler."""
            pipeline = factory.create_pipeline(
                ModelType.SVM, Technique.RANDOM_UNDER_SAMPLING, seed=0
            )

            assert isinstance(pipeline.named_steps["sampler"], RandomUnderSampler)

        def it_creates_pipeline_with_smote_tomek(self, factory: DefaultEstimatorFactory) -> None:
            """Verify SMOTE_TOMEK adds correct sampler."""
            pipeline = factory.create_pipeline(ModelType.SVM, Technique.SMOTE_TOMEK, seed=0)

            assert isinstance(pipeline.named_steps["sampler"], SMOTETomek)

        def it_wraps_classifier_with_metacost(self, factory: DefaultEstimatorFactory) -> None:
            """Verify META_COST wraps classifier."""
            pipeline = factory.create_pipeline(ModelType.SVM, Technique.META_COST, seed=0)

            assert isinstance(pipeline.named_steps["clf"], MetaCostClassifier)

        def it_creates_random_forest_pipeline(self, factory: DefaultEstimatorFactory) -> None:
            """Verify Random Forest model creation."""
            pipeline = factory.create_pipeline(ModelType.RANDOM_FOREST, Technique.BASELINE, seed=0)

            assert isinstance(pipeline.named_steps["clf"], RandomForestClassifier)

        def it_creates_xgboost_pipeline(self, factory: DefaultEstimatorFactory) -> None:
            """Verify XGBoost model creation."""
            pipeline = factory.create_pipeline(ModelType.XGBOOST, Technique.BASELINE, seed=0)

            assert isinstance(pipeline.named_steps["clf"], RobustXGBClassifier)

    class DescribeGetParamGrid:
        """Tests for get_param_grid method."""

        def it_returns_list_of_param_dicts(self, factory: DefaultEstimatorFactory) -> None:
            """Verify returns a list of param dictionaries."""
            grid = factory.get_param_grid(
                ModelType.SVM, Technique.BASELINE, cost_grids=[{0: 1, 1: 2}]
            )

            assert isinstance(grid, list)
            assert len(grid) > 0
            assert isinstance(grid[0], dict)

        def it_includes_svm_hyperparameters(self, factory: DefaultEstimatorFactory) -> None:
            """Verify SVM params include expected keys."""
            grid = factory.get_param_grid(ModelType.SVM, Technique.BASELINE, cost_grids=[])

            assert "clf__C" in grid[0]
            assert "clf__kernel" in grid[0]

        def it_includes_random_forest_hyperparameters(
            self, factory: DefaultEstimatorFactory
        ) -> None:
            """Verify Random Forest params include expected keys."""
            grid = factory.get_param_grid(
                ModelType.RANDOM_FOREST, Technique.BASELINE, cost_grids=[]
            )

            assert "clf__n_estimators" in grid[0]
            assert "clf__max_depth" in grid[0]

        def it_includes_xgboost_hyperparameters(self, factory: DefaultEstimatorFactory) -> None:
            """Verify XGBoost params include expected keys."""
            grid = factory.get_param_grid(ModelType.XGBOOST, Technique.BASELINE, cost_grids=[])

            assert "clf__n_estimators" in grid[0]
            assert "clf__learning_rate" in grid[0]
            assert "clf__max_depth" in grid[0]

        def it_includes_mlp_hyperparameters(self, factory: DefaultEstimatorFactory) -> None:
            """Verify MLP params include expected keys."""
            grid = factory.get_param_grid(ModelType.MLP, Technique.BASELINE, cost_grids=[])

            assert "clf__hidden_layer_sizes" in grid[0]
            assert "clf__activation" in grid[0]

        def it_sets_class_weight_for_cs_svm(self, factory: DefaultEstimatorFactory) -> None:
            """Verify CS-SVM sets class_weight in param grid."""
            cost_matrix = [{0: 1, 1: 2}]
            grid = factory.get_param_grid(ModelType.SVM, Technique.CS_SVM, cost_grids=cost_matrix)

            assert grid[0]["clf__class_weight"] == cost_matrix

        def it_pushes_params_for_metacost(self, factory: DefaultEstimatorFactory) -> None:
            """Verify META_COST pushes params to base_estimator."""
            cost_matrix = [{0: 1, 1: 2}]
            grid = factory.get_param_grid(
                ModelType.SVM, Technique.META_COST, cost_grids=cost_matrix
            )

            assert len(grid) == 1
            assert grid[0]["clf__cost_matrix"] == cost_matrix
            assert "clf__base_estimator__C" in grid[0]


class DescribeGpuSupport:
    """Tests for GPU support in DefaultEstimatorFactory."""

    def it_raises_if_gpu_requested_but_cuml_unavailable(self) -> None:
        """Verify raises ImportError if use_gpu=True but cuML not available."""
        # This test will pass if cuML is not installed
        from experiments.core.modeling.estimators import HAS_CUML

        if not HAS_CUML:
            with pytest.raises(ImportError):
                DefaultEstimatorFactory(use_gpu=True)
