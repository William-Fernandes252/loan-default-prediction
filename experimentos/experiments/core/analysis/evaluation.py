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
