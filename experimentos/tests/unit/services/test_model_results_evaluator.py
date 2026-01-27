"""Tests for model_results_evaluator service."""

import polars as pl

from experiments.core.analysis.metrics import Metric
from experiments.services.model_results_evaluator import ModelResultsEvaluatorImpl

from .conftest import make_model_predictions


class DescribeEvaluate:
    def it_returns_empty_frame_for_no_predictions(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        result = results_evaluator.evaluate(iter([]))  # type: ignore[arg-type]

        collected = result.collect()
        assert collected.shape[0] == 0

    def it_computes_metrics_for_predictions(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        prediction = make_model_predictions(target=[1, 1, 0, 0], prediction=[1, 0, 0, 1])

        result = results_evaluator.evaluate(iter([prediction]))  # type: ignore[arg-type]
        collected = result.collect()

        assert collected.shape[0] == 1
        assert "sensitivity_mean" in collected.columns

    def it_groups_by_model_type_and_technique(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        pred1 = make_model_predictions(seed=1)
        pred2 = make_model_predictions(seed=2, prediction=[1, 1])

        result = results_evaluator.evaluate(iter([pred1, pred2]))  # type: ignore[arg-type]
        collected = result.collect()

        assert collected.shape[0] == 1


class DescribeEvaluatePerSeed:
    def it_returns_empty_frame_for_no_predictions(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        result = results_evaluator.evaluate_per_seed(iter([]))  # type: ignore[arg-type]

        collected = result.collect()
        assert collected.shape[0] == 0

    def it_returns_one_row_per_prediction(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        pred1 = make_model_predictions(seed=1)
        pred2 = make_model_predictions(seed=2, prediction=[0, 0])

        result = results_evaluator.evaluate_per_seed(iter([pred1, pred2]))  # type: ignore[arg-type]
        collected = result.collect()

        assert collected.shape[0] == 2


class DescribeComputeConfusionMatrix:
    def it_computes_tp_tn_fp_fn_correctly(
        self, results_evaluator: ModelResultsEvaluatorImpl, mixed_predictions_lf: pl.LazyFrame
    ) -> None:
        result = results_evaluator._compute_confusion_matrix(mixed_predictions_lf).collect()

        assert result["TP"][0] == 1
        assert result["TN"][0] == 1
        assert result["FP"][0] == 1
        assert result["FN"][0] == 1

    def it_handles_perfect_predictions(
        self, results_evaluator: ModelResultsEvaluatorImpl, perfect_predictions_lf: pl.LazyFrame
    ) -> None:
        result = results_evaluator._compute_confusion_matrix(perfect_predictions_lf).collect()

        assert result["TP"][0] == 2
        assert result["TN"][0] == 2
        assert result["FP"][0] == 0
        assert result["FN"][0] == 0

    def it_handles_all_wrong_predictions(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        predictions = pl.LazyFrame(
            {
                "target": [1, 1, 0, 0],
                "prediction": [0, 0, 1, 1],
            }
        )

        result = results_evaluator._compute_confusion_matrix(predictions).collect()

        assert result["TP"][0] == 0
        assert result["TN"][0] == 0
        assert result["FP"][0] == 2
        assert result["FN"][0] == 2


class DescribeComputeMetrics:
    def it_computes_sensitivity(self, results_evaluator: ModelResultsEvaluatorImpl) -> None:
        confusion = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = results_evaluator._compute_metrics(confusion).collect()

        assert result[Metric.SENSITIVITY][0] == 1.0

    def it_computes_specificity(self, results_evaluator: ModelResultsEvaluatorImpl) -> None:
        confusion = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = results_evaluator._compute_metrics(confusion).collect()

        assert result[Metric.SPECIFICITY][0] == 1.0

    def it_computes_precision(self, results_evaluator: ModelResultsEvaluatorImpl) -> None:
        confusion = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = results_evaluator._compute_metrics(confusion).collect()

        assert result[Metric.PRECISION][0] == 1.0

    def it_fills_zero_for_division_by_zero(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        confusion = pl.LazyFrame({"TP": [0], "TN": [2], "FP": [0], "FN": [2]})

        result = results_evaluator._compute_metrics(confusion).collect()

        assert result[Metric.PRECISION][0] == 0.0

    def it_computes_balanced_accuracy(self, results_evaluator: ModelResultsEvaluatorImpl) -> None:
        confusion = pl.LazyFrame({"TP": [2], "TN": [2], "FP": [0], "FN": [0]})

        result = results_evaluator._compute_metrics(confusion).collect()

        assert result[Metric.BALANCED_ACCURACY][0] == 1.0


class DescribeBuildAggregationExprs:
    def it_returns_mean_and_std_per_metric(
        self, results_evaluator: ModelResultsEvaluatorImpl
    ) -> None:
        exprs = results_evaluator._build_aggregation_exprs(["metric1", "metric2"])

        assert len(exprs) == 4  # 2 metrics × 2 (mean + std)
