"""Implementation of a data repository using a storage backend."""

from dataclasses import dataclass

import polars as pl

from experiments.core.data_new import Dataset
from experiments.storage import Storage


@dataclass(frozen=True, slots=True, kw_only=True)
class DataLayout:
    """Data layout on the storage for data repository operations."""

    raw_data_path_template: str = "data/raw/{dataset_id}.csv"
    interim_data_path_template: str = "data/interim/{dataset_id}.parquet"

    def get_raw_data_key(self, dataset_id: str) -> str:
        """Get the raw data path for a given dataset ID."""
        return self.raw_data_path_template.format(dataset_id=dataset_id)

    def get_interim_data_key(self, dataset_id: str) -> str:
        """Get the interim data path for a given dataset ID."""
        return self.interim_data_path_template.format(dataset_id=dataset_id)


class StorageDataRepository:
    """Repository for data management using a storage backend.

    It implements the DataRepository protocol to fetch and save dataset information using the provided storage system.

    Args:
        storage (Storage): The storage backend to use for data operations.
    """

    def __init__(self, storage: Storage, data_layout: DataLayout) -> None:
        self._storage = storage
        self._data_layout = data_layout

    def get_size_in_bytes(self, dataset: Dataset) -> int:
        key = self._data_layout.get_raw_data_key(dataset.id)
        return self._storage.get_size_bytes(key)

    def get_raw_data(self, dataset: Dataset) -> pl.LazyFrame:
        key = self._data_layout.get_raw_data_key(dataset.id)
        df = self._storage.scan_csv(key, **dataset.get_extra_params())
        return df.lazy()

    def save_interim_data(self, dataset: Dataset, data: pl.DataFrame) -> None:
        key = self._data_layout.get_interim_data_key(dataset.id)
        self._storage.write_parquet(data, key)

    def is_processed(self, dataset: Dataset) -> bool:
        key = self._data_layout.get_interim_data_key(dataset.id)
        return self._storage.exists(key)
