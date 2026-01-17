"""Data processing pipeline base types and context."""

from dataclasses import dataclass
from typing import TypedDict

import polars as pl

from experiments.core.data.datasets import Dataset
from experiments.core.data.repository import DataRepository
from experiments.lib.pipelines import Pipeline, Task

type DataProcessingPipeline = Pipeline[DataProcessingPipelineState, DataProcessingPipelineContext]
"""Data processing pipeline type alias."""


type DataProcessingPipelineTask = Task[DataProcessingPipelineState, DataProcessingPipelineContext]
"""Data processing pipeline task type alias."""


class DataProcessingPipelineState(TypedDict, total=False):
    """State for the data processing pipeline.

    Attributes:
        is_processed: Flag indicating if the data already has been processed.
        raw_data: The raw data loaded from storage.
        interim_data: The interim data after transformations.
        X_final: The final features dataframe.
        y_final: The final target dataframe.
    """

    is_processed: bool
    raw_data: pl.LazyFrame
    interim_data: pl.DataFrame
    X_final: pl.DataFrame
    y_final: pl.DataFrame


@dataclass(frozen=True, slots=True, kw_only=True)
class DataProcessingPipelineContext:
    """Context for data processing pipeline.

    Attributes:
        dataset: The dataset being processed.
        data_repository: The data repository for fetching and saving data.
        use_gpu: Flag to indicate if GPU acceleration should be used.
        force_overwrite: Flag to indicate if existing processed data should be overwritten.
    """

    dataset: Dataset
    data_repository: DataRepository
    use_gpu: bool
    force_overwrite: bool = False
