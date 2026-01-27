"""Defines protocols and types for model results evaluation."""

from typing import Protocol

import polars as pl

from experiments.core.predictions.repository import ModelPredictionsResults

type EvaluationMetrics = pl.LazyFrame
"""Represents the evaluation metrics as lazy frame."""


class ModelResultsEvaluator(Protocol):
    """Protocol for evaluating classification model results."""

    def evaluate(
        self,
        predictions: ModelPredictionsResults,
    ) -> EvaluationMetrics:
        """Evaluates classification results by computing metrics from predictions.

        Args:
            predictions: An iterator of model predictions.

        Returns:
            EvaluationMetrics: A LazyFrame containing the computed evaluation metrics,
                with mean and standard deviation for each metric grouped by model_type
                and technique.
        """
        ...

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
                and all metric values.
        """
        ...
