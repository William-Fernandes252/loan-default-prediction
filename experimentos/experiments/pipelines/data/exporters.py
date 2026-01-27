"""Exporters for processed data from the datasets."""

from experiments.lib.pipelines import TaskResult, TaskStatus
from experiments.pipelines.data.pipeline import (
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)


def export_processed_data_as_parquet(
    state: DataProcessingPipelineState, context: DataProcessingPipelineContext
) -> TaskResult[DataProcessingPipelineState]:
    """Export processed data using the provided exporter.

    Args:
        state: The current pipeline state.
        context: The pipeline context.

    Returns:
        The updated pipeline state after exporting.
    """
    if state["interim_data"] is None:
        return TaskResult(state, TaskStatus.FAILURE, "No interim data found in state to export.")

    context.data_repository.save_interim_data(context.dataset, state["interim_data"])
    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        "Processed data exported successfully.",
    )


def export_final_features_as_parquet(
    state: DataProcessingPipelineState, context: DataProcessingPipelineContext
) -> TaskResult[DataProcessingPipelineState]:
    """Export final features using the provided exporter.

    Args:
        state: The current pipeline state.
        context: The pipeline context.

    Returns:
        The updated pipeline state after exporting.
    """
    if state["X_final"] is None or state["y_final"] is None:
        return TaskResult(state, TaskStatus.FAILURE, "No final features found in state to export.")

    context.data_repository.save_final_features(
        context.dataset, state["X_final"], state["y_final"]
    )
    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        "Final features exported successfully.",
    )
