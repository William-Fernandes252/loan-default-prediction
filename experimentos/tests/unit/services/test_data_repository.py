"""Tests for data_repository service."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.services.data_repository import DataStorageLayout, StorageDataRepository

# ============================================================================
# Helper for creating mock datasets
# ============================================================================


def make_mock_dataset(name: str = "test_dataset") -> MagicMock:
    """Create a mock Dataset with __str__ returning the given name."""
    dataset = MagicMock(spec=Dataset)
    dataset.__str__ = MagicMock(return_value=name)
    return dataset


# ============================================================================
# DataStorageLayout Tests
# ============================================================================


class DescribeDataStorageLayout:
    def it_has_default_raw_template(self, data_layout: DataStorageLayout) -> None:
        assert data_layout.raw_data_key_template == "data/raw/{dataset_id}.csv"

    def it_has_default_interim_template(self, data_layout: DataStorageLayout) -> None:
        assert data_layout.interim_data_key_template == "data/interim/{dataset_id}.parquet"

    def it_has_default_X_final_template(self, data_layout: DataStorageLayout) -> None:
        assert data_layout.X_final_key_template == "data/processed/{dataset_id}_X.parquet"

    def it_has_default_y_final_template(self, data_layout: DataStorageLayout) -> None:
        assert data_layout.y_final_key_template == "data/processed/{dataset_id}_y.parquet"


class DescribeGetRawDataKey:
    def it_formats_key_with_dataset(self, data_layout: DataStorageLayout) -> None:
        dataset = make_mock_dataset("taiwan_credit")

        key = data_layout.get_raw_data_key(dataset)

        assert key == "data/raw/taiwan_credit.csv"


class DescribeGetInterimDataKey:
    def it_formats_key_with_dataset(self, data_layout: DataStorageLayout) -> None:
        dataset = make_mock_dataset("lending_club")

        key = data_layout.get_interim_data_key(dataset)

        assert key == "data/interim/lending_club.parquet"


class DescribeGetFeaturesAndTargetKeys:
    def it_returns_tuple_of_keys(self, data_layout: DataStorageLayout) -> None:
        dataset = make_mock_dataset("corporate_credit")

        X_key, y_key = data_layout.get_features_and_target_keys(dataset)

        assert X_key == "data/processed/corporate_credit_X.parquet"
        assert y_key == "data/processed/corporate_credit_y.parquet"


# ============================================================================
# StorageDataRepository Tests
# ============================================================================


class DescribeStorageDataRepositoryInit:
    def it_stores_dependencies(
        self, mock_storage: MagicMock, data_layout: DataStorageLayout
    ) -> None:
        repo = StorageDataRepository(mock_storage, data_layout)

        assert repo._storage is mock_storage
        assert repo._data_layout is data_layout


class DescribeGetSizeInBytes:
    def it_returns_size_from_storage(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_storage.get_size_bytes.return_value = 1024

        result = data_repository.get_size_in_bytes(dataset)

        assert result == 1024
        mock_storage.get_size_bytes.assert_called_once_with("data/raw/test_dataset.csv")


class DescribeGetRawData:
    def it_scans_csv_from_storage(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = MagicMock()
        dataset.__str__ = MagicMock(return_value="test_dataset")
        dataset.get_extra_params = MagicMock(return_value={"separator": ","})
        mock_df = MagicMock()
        mock_df.lazy.return_value = MagicMock()
        mock_storage.scan_csv.return_value = mock_df

        data_repository.get_raw_data(dataset)

        mock_storage.scan_csv.assert_called_once_with("data/raw/test_dataset.csv", separator=",")


class DescribeSaveInterimData:
    def it_writes_parquet_to_storage(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        data = pl.DataFrame({"col": [1, 2, 3]})

        data_repository.save_interim_data(dataset, data)

        call_args = mock_storage.write_parquet.call_args
        assert call_args[0][1] == "data/interim/test_dataset.parquet"


class DescribeGetInterimData:
    def it_scans_parquet_from_storage(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_lazy = MagicMock()
        mock_storage.scan_parquet.return_value = mock_lazy

        result = data_repository.get_interim_data(dataset)

        mock_storage.scan_parquet.assert_called_once_with("data/interim/test_dataset.parquet")
        assert result is mock_lazy


class DescribeIsProcessed:
    def it_returns_true_when_both_files_exist(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_storage.exists.side_effect = [True, True]

        result = data_repository.is_processed(dataset)

        assert result is True
        assert mock_storage.exists.call_count == 2

    def it_returns_false_when_X_missing(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_storage.exists.side_effect = [False, True]

        result = data_repository.is_processed(dataset)

        assert result is False

    def it_returns_false_when_y_missing(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_storage.exists.side_effect = [True, False]

        result = data_repository.is_processed(dataset)

        assert result is False


class DescribeSaveFinalFeatures:
    def it_writes_both_X_and_y_as_parquet(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        X = pl.DataFrame({"feature": [1, 2]})
        y = pl.DataFrame({"target": [0, 1]})

        data_repository.save_final_features(dataset, X, y)

        assert mock_storage.write_parquet.call_count == 2


class DescribeLoadTrainingData:
    def it_returns_training_data_when_files_exist(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_X = MagicMock()
        mock_y = MagicMock()
        mock_storage.read_parquet.side_effect = [mock_X, mock_y]

        result = data_repository.load_training_data(dataset)

        assert result.X is mock_X
        assert result.y is mock_y

    def it_raises_value_error_when_files_not_found(
        self, mock_storage: MagicMock, data_repository: StorageDataRepository
    ) -> None:
        dataset = make_mock_dataset()
        mock_storage.read_parquet.side_effect = FileNotFoundError("Not found")

        with pytest.raises(ValueError) as exc_info:
            data_repository.load_training_data(dataset)

        assert "Training data not found" in str(exc_info.value)
