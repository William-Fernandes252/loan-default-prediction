"""Data loading tasks for data processing pipelines."""

from experiments.lib.pipelines import TaskResult, TaskStatus
from experiments.pipelines.data.pipeline import (
    DataProcessingPipelineContext,
    DataProcessingPipelineState,
)


def load_raw_data_from_csv(
    state: DataProcessingPipelineState, context: DataProcessingPipelineContext
) -> TaskResult[DataProcessingPipelineState]:
    """Load raw data from CSV and update the pipeline state.

    Args:
        state: The current pipeline state.
        context: The data pipeline context.

    Returns:
        The updated pipeline state with raw data loaded.
    """
    state["raw_data"] = context.data_repository.get_raw_data(context.dataset)
    return TaskResult(state, TaskStatus.SUCCESS, "Raw data loaded successfully.")
