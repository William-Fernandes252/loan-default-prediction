"""Tests for model_results_evaluator service."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.analysis.metrics import Metric
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import ModelPredictions
from experiments.services.model_results_evaluator import ModelResultsEvaluatorImpl


class DescribeModelResultsEvaluatorImplEvaluate:
    @pytest.fixture
    def evaluator(self) -> ModelResultsEvaluatorImpl:
        return ModelResultsEvaluatorImpl()

    def it_returns_empty_lazyframe_when_no_predictions(
        self, evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        result = evaluator.evaluate(iter([]))  # type: ignore[arg-type]

        collected = result.collect()
        assert collected.shape[0] == 0

    def it_computes_metrics_for_predictions(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        predictions_lf = pl.LazyFrame(
            {
                "target": [1, 1, 0, 0],
                "prediction": [1, 0, 0, 1],
            }
        )
        prediction = ModelPredictions(
            execution_id="exec-1",
            seed=42,
            dataset=MagicMock(),
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            predictions=predictions_lf,
        )

        result = evaluator.evaluate(iter([prediction]))  # type: ignore[arg-type]
        collected = result.collect()

        assert collected.shape[0] == 1
        assert "sensitivity_mean" in collected.columns

    def it_groups_by_model_type_and_technique(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        pred1 = ModelPredictions(
            execution_id="exec-1",
            seed=1,
            dataset=MagicMock(),
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            predictions=pl.LazyFrame({"target": [1, 0], "prediction": [1, 0]}),
        )
        pred2 = ModelPredictions(
            execution_id="exec-1",
            seed=2,
            dataset=MagicMock(),
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            predictions=pl.LazyFrame({"target": [1, 0], "prediction": [1, 1]}),
        )

        result = evaluator.evaluate(iter([pred1, pred2]))  # type: ignore[arg-type]
        collected = result.collect()

        # Should have one row (grouped by model_type, technique)
        assert collected.shape[0] == 1


class DescribeEvaluatePerSeed:
    @pytest.fixture
    def evaluator(self) -> ModelResultsEvaluatorImpl:
        return ModelResultsEvaluatorImpl()

    def it_returns_empty_lazyframe_when_no_predictions(
        self, evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        result = evaluator.evaluate_per_seed(iter([]))  # type: ignore[arg-type]

        collected = result.collect()
        assert collected.shape[0] == 0

    def it_returns_one_row_per_prediction(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        pred1 = ModelPredictions(
            execution_id="exec-1",
            seed=1,
            dataset=MagicMock(),
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            predictions=pl.LazyFrame({"target": [1, 0], "prediction": [1, 0]}),
        )
        pred2 = ModelPredictions(
            execution_id="exec-1",
            seed=2,
            dataset=MagicMock(),
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            predictions=pl.LazyFrame({"target": [1, 0], "prediction": [0, 0]}),
        )

        result = evaluator.evaluate_per_seed(iter([pred1, pred2]))  # type: ignore[arg-type]
        collected = result.collect()

        # Should have two rows (one per prediction)
        assert collected.shape[0] == 2


class DescribeComputeConfusionMatrix:
    @pytest.fixture
    def evaluator(self) -> ModelResultsEvaluatorImpl:
        return ModelResultsEvaluatorImpl()

    def it_computes_correct_tp_tn_fp_fn(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        # TP=1, TN=1, FP=1, FN=1
        predictions_lf = pl.LazyFrame(
            {
                "target": [1, 1, 0, 0],
                "prediction": [1, 0, 0, 1],
            }
        )

        result = evaluator._compute_confusion_matrix(predictions_lf).collect()

        assert result["TP"][0] == 1
        assert result["TN"][0] == 1
        assert result["FP"][0] == 1
        assert result["FN"][0] == 1

    def it_handles_perfect_predictions(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        predictions_lf = pl.LazyFrame(
            {
                "target": [1, 1, 0, 0],
                "prediction": [1, 1, 0, 0],
            }
        )

        result = evaluator._compute_confusion_matrix(predictions_lf).collect()

        assert result["TP"][0] == 2
        assert result["TN"][0] == 2
        assert result["FP"][0] == 0
        assert result["FN"][0] == 0

    def it_handles_all_wrong_predictions(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        predictions_lf = pl.LazyFrame(
            {
                "target": [1, 1, 0, 0],
                "prediction": [0, 0, 1, 1],
            }
        )

        result = evaluator._compute_confusion_matrix(predictions_lf).collect()

        assert result["TP"][0] == 0
        assert result["TN"][0] == 0
        assert result["FP"][0] == 2
        assert result["FN"][0] == 2


class DescribeComputeMetrics:
    @pytest.fixture
    def evaluator(self) -> ModelResultsEvaluatorImpl:
        return ModelResultsEvaluatorImpl()

    def it_computes_sensitivity(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        # TP=2, FN=0 -> sensitivity = 1.0
        confusion_lf = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = evaluator._compute_metrics(confusion_lf).collect()

        assert result[Metric.SENSITIVITY][0] == 1.0

    def it_computes_specificity(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        # TN=2, FP=0 -> specificity = 1.0
        confusion_lf = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = evaluator._compute_metrics(confusion_lf).collect()

        assert result[Metric.SPECIFICITY][0] == 1.0

    def it_computes_precision(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        # TP=2, FP=0 -> precision = 1.0
        confusion_lf = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = evaluator._compute_metrics(confusion_lf).collect()

        assert result[Metric.PRECISION][0] == 1.0

    def it_handles_zero_division_with_nan_fill(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        # TP=0, FP=0 -> precision would be 0/0, filled with 0.0
        confusion_lf = pl.LazyFrame({"TP": [0], "TN": [2], "FP": [0], "FN": [2]})

        result = evaluator._compute_metrics(confusion_lf).collect()

        assert result[Metric.PRECISION][0] == 0.0

    def it_computes_balanced_accuracy(self, evaluator: ModelResultsEvaluatorImpl) -> None:
        # sensitivity=1.0, specificity=1.0 -> balanced_accuracy = 1.0
        confusion_lf = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = evaluator._compute_metrics(confusion_lf).collect()

        assert result[Metric.BALANCED_ACCURACY][0] == 1.0


class DescribeBuildAggregationExprs:
    @pytest.fixture
    def evaluator(self) -> ModelResultsEvaluatorImpl:
        return ModelResultsEvaluatorImpl()

    def it_returns_mean_and_std_for_each_metric(
        self, evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        metric_ids = ["metric1", "metric2"]

        exprs = evaluator._build_aggregation_exprs(metric_ids)

        # Should have 2 expressions per metric (mean and std)
        assert len(exprs) == 4
