from typing import Annotated, TypedDict

import polars as pl


class DataPipelineState(TypedDict):
    """State for the data processing pipeline."""

    is_processed: Annotated[bool, "Flag indicating if the data already has been processed."]
    raw_data: Annotated[pl.LazyFrame | None, "The raw data loaded from storage."]
    interim_data: Annotated[pl.DataFrame | None, "The interim data after transformations."]
    X_final: Annotated[pl.DataFrame | None, "The final features dataframe."]
    y_final: Annotated[pl.DataFrame | None, "The final target dataframe."]


_default_initial_state: DataPipelineState = {
    "raw_data": None,
    "interim_data": None,
    "X_final": None,
    "y_final": None,
    "is_processed": False,
}


def get_default_initial_state() -> DataPipelineState:
    """Returns a copy of the default initial state for the data processing pipeline."""
    return _default_initial_state.copy()
