"""Tests for data_repository service."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.services.data_repository import DataStorageLayout, StorageDataRepository


class DescribeDataStorageLayout:
    @pytest.fixture
    def layout(self) -> DataStorageLayout:
        return DataStorageLayout()

    def it_has_default_raw_template(self, layout: DataStorageLayout) -> None:
        assert layout.raw_data_key_template == "data/raw/{dataset_id}.csv"

    def it_has_default_interim_template(self, layout: DataStorageLayout) -> None:
        assert layout.interim_data_key_template == "data/interim/{dataset_id}.parquet"

    def it_has_default_X_final_template(self, layout: DataStorageLayout) -> None:
        assert layout.X_final_key_template == "data/processed/{dataset_id}_X.parquet"

    def it_has_default_y_final_template(self, layout: DataStorageLayout) -> None:
        assert layout.y_final_key_template == "data/processed/{dataset_id}_y.parquet"


class DescribeGetRawDataKey:
    def it_formats_key_with_dataset(self) -> None:
        layout = DataStorageLayout()
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="taiwan_credit")

        key = layout.get_raw_data_key(dataset)

        assert key == "data/raw/taiwan_credit.csv"


class DescribeGetInterimDataKey:
    def it_formats_key_with_dataset(self) -> None:
        layout = DataStorageLayout()
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="lending_club")

        key = layout.get_interim_data_key(dataset)

        assert key == "data/interim/lending_club.parquet"


class DescribeGetFeaturesAndTargetKeys:
    def it_returns_tuple_of_keys(self) -> None:
        layout = DataStorageLayout()
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="corporate_credit")

        X_key, y_key = layout.get_features_and_target_keys(dataset)

        assert X_key == "data/processed/corporate_credit_X.parquet"
        assert y_key == "data/processed/corporate_credit_y.parquet"


class DescribeStorageDataRepositoryInit:
    def it_stores_storage_backend(self) -> None:
        storage = MagicMock()
        layout = DataStorageLayout()

        repo = StorageDataRepository(storage, layout)

        assert repo._storage is storage

    def it_stores_data_layout(self) -> None:
        storage = MagicMock()
        layout = DataStorageLayout()

        repo = StorageDataRepository(storage, layout)

        assert repo._data_layout is layout


class DescribeGetSizeInBytes:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_returns_size_from_storage(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        storage.get_size_bytes.return_value = 1024

        result = repo.get_size_in_bytes(dataset)

        assert result == 1024
        storage.get_size_bytes.assert_called_once_with("data/raw/test_dataset.csv")


class DescribeGetRawData:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_scans_csv_from_storage(self, storage: MagicMock, repo: StorageDataRepository) -> None:
        dataset = MagicMock()
        dataset.__str__ = MagicMock(return_value="test_dataset")
        dataset.get_extra_params = MagicMock(return_value={"separator": ","})
        mock_df = MagicMock()
        mock_lazy = MagicMock()
        mock_df.lazy.return_value = mock_lazy
        storage.scan_csv.return_value = mock_df

        repo.get_raw_data(dataset)

        storage.scan_csv.assert_called_once_with("data/raw/test_dataset.csv", separator=",")


class DescribeSaveInterimData:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_writes_parquet_to_storage(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        data = pl.DataFrame({"col": [1, 2, 3]})

        repo.save_interim_data(dataset, data)

        storage.write_parquet.assert_called_once()
        call_args = storage.write_parquet.call_args
        assert call_args[0][1] == "data/interim/test_dataset.parquet"


class DescribeGetInterimData:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_scans_parquet_from_storage(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        mock_lazy = MagicMock()
        storage.scan_parquet.return_value = mock_lazy

        result = repo.get_interim_data(dataset)

        storage.scan_parquet.assert_called_once_with("data/interim/test_dataset.parquet")
        assert result is mock_lazy


class DescribeIsProcessed:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_returns_true_when_both_files_exist(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        storage.exists.side_effect = [True, True]

        result = repo.is_processed(dataset)

        assert result is True
        assert storage.exists.call_count == 2

    def it_returns_false_when_X_missing(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        storage.exists.side_effect = [False, True]

        result = repo.is_processed(dataset)

        assert result is False

    def it_returns_false_when_y_missing(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        storage.exists.side_effect = [True, False]

        result = repo.is_processed(dataset)

        assert result is False


class DescribeSaveFinalFeatures:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_writes_both_X_and_y_as_parquet(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        X = pl.DataFrame({"feature": [1, 2]})
        y = pl.DataFrame({"target": [0, 1]})

        repo.save_final_features(dataset, X, y)

        assert storage.write_parquet.call_count == 2


class DescribeLoadTrainingData:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> StorageDataRepository:
        return StorageDataRepository(storage, DataStorageLayout())

    def it_returns_training_data_when_files_exist(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        mock_X = MagicMock()
        mock_y = MagicMock()
        storage.scan_parquet.side_effect = [mock_X, mock_y]

        result = repo.load_training_data(dataset)

        assert result.X is mock_X
        assert result.y is mock_y

    def it_raises_value_error_when_files_not_found(
        self, storage: MagicMock, repo: StorageDataRepository
    ) -> None:
        dataset = MagicMock(spec=Dataset)
        dataset.__str__ = MagicMock(return_value="test_dataset")
        storage.scan_parquet.side_effect = FileNotFoundError("Not found")

        with pytest.raises(ValueError) as exc_info:
            repo.load_training_data(dataset)

        assert "Training data not found" in str(exc_info.value)
