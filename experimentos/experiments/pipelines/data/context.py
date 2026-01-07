from dataclasses import dataclass
from typing import Annotated

from experiments.core.data import Dataset
from experiments.core.data.repository import DataRepository


@dataclass(frozen=True, slots=True, kw_only=True)
class DataPipelineContext:
    """Context for data processing pipeline."""

    dataset: Annotated[Dataset, "The dataset being processed."]
    data_repository: Annotated[DataRepository, "The data repository for fetching and saving data."]
    use_gpu: Annotated[bool, "Flag to indicate if GPU acceleration should be used."]
