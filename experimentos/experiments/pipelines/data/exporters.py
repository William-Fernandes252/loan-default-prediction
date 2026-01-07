"""Exporters for processed data from the datasets."""

from experiments.lib.pipelines import TaskResult, TaskStatus
from experiments.pipelines.data.context import DataPipelineContext
from experiments.pipelines.data.state import DataPipelineState


def export_processed_data_as_parquet(
    state: DataPipelineState, context: DataPipelineContext
) -> TaskResult[DataPipelineState]:
    """Export processed data using the provided exporter.

    Args:
        state: The current pipeline state.
        exporter: The exporter instance to use for exporting.

    Returns:
        The updated pipeline state after exporting.
    """
    if state["interim_data"] is None:
        return TaskResult(state, TaskStatus.FAILURE, "No interim data found in state to export.")

    try:
        context.data_repository.save_interim_data(context.dataset, state["interim_data"])
        return TaskResult(
            state,
            TaskStatus.SUCCESS,
            f"Processed data for the dataset {context.dataset.id} exported successfully.",
        )
    except Exception as e:
        return TaskResult(
            state,
            TaskStatus.ERROR,
            f"Failed to export processed data as parquet for dataset {context.dataset.id}",
            e,
        )


def export_final_features_as_parquet(
    state: DataPipelineState, context: DataPipelineContext
) -> TaskResult[DataPipelineState]:
    """Export final features using the provided exporter.

    Args:
        state: The current pipeline state.
        context: The pipeline context.

    Returns:
        The updated pipeline state after exporting.
    """
    if state["X_final"] is None or state["y_final"] is None:
        return TaskResult(state, TaskStatus.FAILURE, "No final features found in state to export.")

    try:
        context.data_repository.save_final_features(
            context.dataset, state["X_final"], state["y_final"]
        )
    except Exception as e:
        return TaskResult(
            state,
            TaskStatus.ERROR,
            f"Failed to export final features as parquet for dataset {context.dataset.id}",
            e,
        )
    return TaskResult(
        state,
        TaskStatus.SUCCESS,
        f"Final features for the dataset {context.dataset.id} exported successfully.",
    )
