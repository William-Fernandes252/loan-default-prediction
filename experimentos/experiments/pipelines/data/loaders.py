from experiments.lib.pipelines import TaskResult, TaskStatus
from experiments.pipelines.data.context import DataPipelineContext
from experiments.pipelines.data.state import DataPipelineState


def load_raw_data_from_csv(
    state: DataPipelineState, context: DataPipelineContext
) -> TaskResult[DataPipelineState]:
    """Load raw data from CSV and update the pipeline state.

    Args:
        state: The current pipeline state.
        context: The data pipeline context.

    Returns:
        The updated pipeline state with raw data loaded.
    """
    try:
        state["raw_data"] = context.data_repository.get_raw_data(context.dataset)
        return TaskResult(state, TaskStatus.SUCCESS, "Raw data loaded successfully.")
    except Exception as e:
        return TaskResult(
            state,
            TaskStatus.ERROR,
            f"Failed to load raw data for dataset {context.dataset.id}",
            e,
        )
