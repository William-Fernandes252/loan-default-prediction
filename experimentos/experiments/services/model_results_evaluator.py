"""Implementation of the model results evaluation process."""

from functools import reduce

import polars as pl

from experiments.core.analysis.evaluation import EvaluationMetrics
from experiments.core.analysis.metrics import Metric
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import ModelPredictions, ModelPredictionsResults


class ModelResultsEvaluatorImpl:
    """Evaluates classification model results by computing metrics from predictions.

    This evaluator computes confusion matrix-based metrics for each model prediction
    and can provide either aggregated metrics (mean/std) or per-seed metrics for
    stability analysis.
    """

    def evaluate(
        self,
        predictions: ModelPredictionsResults,
    ) -> EvaluationMetrics:
        """Evaluates classification results by computing metrics from predictions.

        Computes the following metrics for each prediction:
        - Sensitivity (Recall/TPR)
        - Specificity (TNR)
        - Precision
        - F1 Score
        - G-Mean
        - Balanced Accuracy

        Results are aggregated by model_type and technique with mean and standard
        deviation for each metric.

        Args:
            predictions: An iterator of model predictions.

        Returns:
            EvaluationMetrics: A LazyFrame containing the computed evaluation metrics,
                with mean and standard deviation for each metric grouped by model_type
                and technique.
        """
        # Compute metrics for each prediction
        metrics_list = [self._compute_metrics_for_prediction(p) for p in predictions]

        if not metrics_list:
            return pl.LazyFrame()

        # Concatenate all metrics and aggregate by model_type and technique
        metric_ids = [m.value for m in Metric]
        return (
            reduce(lambda acc, lf: pl.concat([acc, lf]), metrics_list)
            .group_by("model_type", "technique")
            .agg(*self._build_aggregation_exprs(metric_ids))
        )

    def evaluate_per_seed(
        self,
        predictions: ModelPredictionsResults,
    ) -> pl.LazyFrame:
        """Evaluates classification results without aggregation.

        Computes per-seed metrics for each model/technique combination. This is
        useful for stability analysis where the distribution across seeds needs
        to be visualized (e.g., boxplots).

        Args:
            predictions: An iterator of model predictions.

        Returns:
            pl.LazyFrame: A LazyFrame containing computed metrics for each
                individual prediction, with columns: model_type, technique,
                seed, and all metric values.
        """
        metrics_list = [self._compute_metrics_for_prediction(p) for p in predictions]

        if not metrics_list:
            return pl.LazyFrame()

        return reduce(lambda acc, lf: pl.concat([acc, lf]), metrics_list)

    def _compute_confusion_matrix(self, predictions_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Computes the confusion matrix components from binary predictions.

        Args:
            predictions_lf: A LazyFrame with "target" and "prediction" columns (0 or 1).

        Returns:
            A LazyFrame with a single row containing TP, TN, FP, FN counts.
        """
        return predictions_lf.select(
            pl.when((pl.col("target") == 1) & (pl.col("prediction") == 1))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("TP"),
            pl.when((pl.col("target") == 0) & (pl.col("prediction") == 0))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("TN"),
            pl.when((pl.col("target") == 0) & (pl.col("prediction") == 1))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("FP"),
            pl.when((pl.col("target") == 1) & (pl.col("prediction") == 0))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("FN"),
        )

    def _compute_metrics(self, confusion_matrix_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Computes all classification metrics from confusion matrix components.

        Metrics are computed in dependency order: base metrics first (sensitivity,
        specificity, precision), then composite metrics (F1, G-Mean, Balanced Accuracy).

        Args:
            confusion_matrix_lf: A LazyFrame with TP, TN, FP, FN columns.

        Returns:
            A LazyFrame with all computed metric columns.
        """
        # Base metrics (no dependencies)
        sensitivity_expr = (pl.col("TP") / (pl.col("TP") + pl.col("FN"))).fill_nan(0.0)
        specificity_expr = (pl.col("TN") / (pl.col("TN") + pl.col("FP"))).fill_nan(0.0)
        precision_expr = (pl.col("TP") / (pl.col("TP") + pl.col("FP"))).fill_nan(0.0)

        # Composite metrics (depend on base metrics)
        f1_expr = (
            2
            * (pl.col(Metric.PRECISION) * pl.col(Metric.SENSITIVITY))
            / (pl.col(Metric.PRECISION) + pl.col(Metric.SENSITIVITY))
        ).fill_nan(0.0)

        g_mean_expr = (
            (pl.col(Metric.SENSITIVITY) * pl.col(Metric.SPECIFICITY)).sqrt().fill_nan(0.0)
        )

        balanced_accuracy_expr = (pl.col(Metric.SENSITIVITY) + pl.col(Metric.SPECIFICITY)) / 2

        return confusion_matrix_lf.with_columns(
            # Base metrics
            sensitivity_expr.alias(Metric.SENSITIVITY),
            specificity_expr.alias(Metric.SPECIFICITY),
            precision_expr.alias(Metric.PRECISION),
        ).with_columns(
            # Composite metrics (computed after base metrics are available)
            f1_expr.alias(Metric.F1_SCORE),
            g_mean_expr.alias(Metric.G_MEAN),
            balanced_accuracy_expr.alias(Metric.BALANCED_ACCURACY),
        )

    def _compute_metrics_for_prediction(
        self,
        model_prediction: ModelPredictions,
    ) -> pl.LazyFrame:
        """Computes metrics for a single model prediction.

        Args:
            model_prediction: The model prediction containing predictions LazyFrame.

        Returns:
            A LazyFrame with computed metrics and model/technique identifiers.
        """
        return (
            model_prediction.predictions.pipe(self._compute_confusion_matrix)
            .pipe(self._compute_metrics)
            .with_columns(
                pl.lit(model_prediction.model_type).cast(pl.Enum(ModelType)).alias("model_type"),
                pl.lit(model_prediction.technique).cast(pl.Enum(Technique)).alias("technique"),
            )
        )

    def _build_aggregation_exprs(self, metric_ids: list[str]) -> list[pl.Expr]:
        """Builds aggregation expressions for mean and std of each metric.

        Args:
            metric_ids: List of metric column names.

        Returns:
            List of Polars expressions for mean and std aggregations.
        """
        return [
            expr
            for metric_id in metric_ids
            for expr in (
                pl.col(metric_id).mean().alias(f"{metric_id}_mean"),
                pl.col(metric_id).std().alias(f"{metric_id}_std"),
            )
        ]
