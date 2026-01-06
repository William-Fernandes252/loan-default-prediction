from typing import Annotated, TypedDict

import polars as pl


class DataPipelineState(TypedDict):
    """State for the data processing pipeline."""

    raw_data: Annotated[pl.LazyFrame, "The raw data loaded from storage."]
    interim_data: Annotated[pl.DataFrame, "The interim data after transformations."]
