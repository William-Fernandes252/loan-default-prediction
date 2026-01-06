"""Exporters for processed data from the datasets."""

from experiments.lib.pipelines import PipelineException
from experiments.pipelines.data.context import DataPipelineContext
from experiments.pipelines.data.state import DataPipelineState


def export_processed_data_as_parquet(
    state: DataPipelineState, context: DataPipelineContext
) -> DataPipelineState:
    """Export processed data using the provided exporter.

    Args:
        state: The current pipeline state.
        exporter: The exporter instance to use for exporting.

    Returns:
        The updated pipeline state after exporting.
    """
    if "interim_data" not in state:
        raise PipelineException("No interim data found in state to export.")

    try:
        context.data_repository.save_interim_data(context.dataset, state["interim_data"])
    except Exception as e:
        raise PipelineException(
            f"Failed to export processed data as parquet for dataset {context.dataset.id}: {e}"
        ) from e
    return state
