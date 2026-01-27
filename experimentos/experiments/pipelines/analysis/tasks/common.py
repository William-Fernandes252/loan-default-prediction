"""Common tasks for loading experiment results and computing metrics.

These tasks form the initial stages of analysis pipelines, handling data
loading from the predictions repository and computing evaluation metrics.
"""

import polars as pl

from experiments.lib.pipelines.tasks import TaskResult, TaskStatus
from experiments.pipelines.analysis.pipeline import (
    AnalysisPipelineContext,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)


def load_experiment_results[T](
    state: AnalysisPipelineState[T],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[T]:
    """Load experiment results (predictions) from the repository.

    Fetches the latest model predictions for the dataset specified in the
    context and stores them in the pipeline state.

    Args:
        state: The current state of the analysis pipeline.
        context: The context containing the predictions repository and dataset.

    Returns:
        AnalysisPipelineTaskResult: Updated state with loaded predictions,
        or failure status if no predictions are found.
    """
    predictions = context.predictions_repository.get_latest_predictions_for_experiment(
        context.dataset
    )

    if predictions is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            f"No predictions found for dataset {context.dataset.value}.",
        )

    updated_state: AnalysisPipelineState[T] = {**state, "model_predictions": predictions}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Loaded predictions for dataset {context.dataset.value}.",
    )


def compute_metrics[T](
    state: AnalysisPipelineState[T],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[T]:
    """Compute evaluation metrics from loaded model predictions.

    Uses the results evaluator from the context to compute metrics from
    the predictions stored in the state. The result is kept as a LazyFrame
    to defer computation until needed.

    Args:
        state: The current state containing model predictions.
        context: The context containing the results evaluator.

    Returns:
        AnalysisPipelineTaskResult: Updated state with computed metrics
        as a LazyFrame, or failure status if no predictions are available.
    """
    model_predictions = state.get("model_predictions")

    if model_predictions is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No model predictions available to compute metrics.",
        )

    metrics: pl.LazyFrame = context.results_evaluator.evaluate(model_predictions)

    updated_state: AnalysisPipelineState[T] = {**state, "metrics": metrics}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        "Computed evaluation metrics (lazy).",
    )


# Task aliases for explicit naming in pipelines
LoadExperimentResultsTask = load_experiment_results
"""Task to load experiment results from the predictions repository."""

ComputeMetricsTask = compute_metrics
"""Task to compute evaluation metrics from model predictions."""


__all__ = [
    "LoadExperimentResultsTask",
    "ComputeMetricsTask",
    "load_experiment_results",
    "compute_metrics",
]
