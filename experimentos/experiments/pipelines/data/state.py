from typing import Annotated, TypedDict

import polars as pl


class DataPipelineState(TypedDict):
    """State for the data processing pipeline."""

    raw_data: Annotated[pl.LazyFrame | None, "The raw data loaded from storage."]
    interim_data: Annotated[pl.DataFrame | None, "The interim data after transformations."]
    X_final: Annotated[pl.DataFrame | None, "The final features dataframe."]
    y_final: Annotated[pl.DataFrame | None, "The final target dataframe."]
