"""Implementation of a data repository using a storage backend."""

from dataclasses import dataclass

import polars as pl

from experiments.core.data import Dataset
from experiments.storage import Storage


@dataclass(frozen=True, slots=True, kw_only=True)
class DataLayout:
    """Data layout on the storage for data repository operations."""

    raw_data_key_template: str = "data/raw/{dataset_id}.csv"
    interim_data_key_template: str = "data/interim/{dataset_id}.parquet"
    X_final_key_template: str = "data/processed/{dataset_id}_X.parquet"
    y_final_key_template: str = "data/processed/{dataset_id}_y.parquet"

    def get_raw_data_key(self, dataset: Dataset) -> str:
        """Get the raw data key for a given dataset ID."""
        return self.raw_data_key_template.format(dataset_id=dataset.value)

    def get_interim_data_key(self, dataset: Dataset) -> str:
        """Get the interim data key for a given dataset ID."""
        return self.interim_data_key_template.format(dataset_id=dataset.value)

    def get_features_and_target_keys(self, dataset: Dataset) -> tuple[str, str]:
        """Get the final features (X) key for a given dataset ID."""
        return self.X_final_key_template.format(
            dataset_id=dataset.value
        ), self.y_final_key_template.format(dataset_id=dataset.value)


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
        key = self._data_layout.get_raw_data_key(dataset)
        return self._storage.get_size_bytes(key)

    def get_raw_data(self, dataset: Dataset) -> pl.LazyFrame:
        key = self._data_layout.get_raw_data_key(dataset)
        df = self._storage.scan_csv(key, **dataset.get_extra_params())
        return df.lazy()

    def save_interim_data(self, dataset: Dataset, data: pl.DataFrame) -> None:
        key = self._data_layout.get_interim_data_key(dataset)
        self._storage.write_parquet(data, key)

    def get_interim_data(self, dataset: Dataset) -> pl.LazyFrame:
        key = self._data_layout.get_interim_data_key(dataset)
        df = self._storage.scan_parquet(key)
        return df

    def is_processed(self, dataset: Dataset) -> bool:
        X_key, y_key = self._data_layout.get_features_and_target_keys(dataset)
        return self._storage.exists(X_key) and self._storage.exists(y_key)

    def save_final_features(self, dataset: Dataset, X: pl.DataFrame, y: pl.DataFrame) -> None:
        X_key, y_key = self._data_layout.get_features_and_target_keys(dataset)
        self._storage.write_parquet(X, X_key)
        self._storage.write_parquet(y, y_key)
