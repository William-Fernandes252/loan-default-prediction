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
    """Compute aggregated evaluation metrics from loaded model predictions.

    Uses the results evaluator from the context to compute metrics from
    the predictions stored in the state. The result is aggregated by
    model_type and technique with mean/std for each metric.

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
        "Computed aggregated evaluation metrics (lazy).",
    )


def compute_per_seed_metrics[T](
    state: AnalysisPipelineState[T],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[T]:
    """Compute per-seed evaluation metrics for stability analysis.

    Uses the results evaluator to compute non-aggregated metrics for each
    individual prediction (seed). This is useful for visualizations that
    require the full distribution, such as boxplots.

    Args:
        state: The current state containing model predictions.
        context: The context containing the results evaluator.

    Returns:
        AnalysisPipelineTaskResult: Updated state with per_seed_metrics
        as a LazyFrame, or failure status if no predictions are available.
    """
    model_predictions = state.get("model_predictions")

    if model_predictions is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No model predictions available to compute per-seed metrics.",
        )

    per_seed_metrics: pl.LazyFrame = context.results_evaluator.evaluate_per_seed(model_predictions)

    updated_state: AnalysisPipelineState[T] = {**state, "per_seed_metrics": per_seed_metrics}
    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        "Computed per-seed evaluation metrics (lazy).",
    )


# Task aliases for explicit naming in pipelines
LoadExperimentResultsTask = load_experiment_results
"""Task to load experiment results from the predictions repository."""

ComputeMetricsTask = compute_metrics
"""Task to compute aggregated evaluation metrics from model predictions."""

ComputePerSeedMetricsTask = compute_per_seed_metrics
"""Task to compute per-seed evaluation metrics for stability analysis."""


__all__ = [
    "LoadExperimentResultsTask",
    "ComputeMetricsTask",
    "ComputePerSeedMetricsTask",
    "load_experiment_results",
    "compute_metrics",
    "compute_per_seed_metrics",
]
