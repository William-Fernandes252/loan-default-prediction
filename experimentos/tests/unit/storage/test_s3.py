from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
import polars as pl
import pytest

from experiments.storage.errors import FileDoesNotExistError, StorageError
from experiments.storage.interface import FileInfo
from experiments.storage.s3 import S3Storage


class DescribeS3StorageExists:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_returns_true_when_object_exists(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        mock_s3_client.head_object.return_value = {"ContentLength": 100}

        result = storage.exists("data/file.txt")

        assert result is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="data/file.txt"
        )

    def it_returns_false_when_object_does_not_exist(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        error = ClientError({"Error": {"Code": "404"}}, "head_object")
        mock_s3_client.head_object.side_effect = error

        result = storage.exists("missing.txt")

        assert result is False

    def it_raises_storage_error_on_other_errors(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        error = ClientError({"Error": {"Code": "AccessDenied"}}, "head_object")
        mock_s3_client.head_object.side_effect = error

        with pytest.raises(StorageError):
            storage.exists("forbidden.txt")


class DescribeS3StorageDelete:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_deletes_existing_object(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        mock_s3_client.head_object.return_value = {"ContentLength": 100}

        storage.delete("data/to_delete.txt")

        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="data/to_delete.txt"
        )

    def it_raises_error_when_object_does_not_exist(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        error = ClientError({"Error": {"Code": "404"}}, "head_object")
        mock_s3_client.head_object.side_effect = error

        with pytest.raises(FileDoesNotExistError) as exc_info:
            storage.delete("missing.txt")

        assert exc_info.value.uri == "missing.txt"


class DescribeS3StorageListFiles:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "data/file1.csv",
                        "Size": 1024,
                        "LastModified": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    },
                    {
                        "Key": "data/file2.csv",
                        "Size": 2048,
                        "LastModified": datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                    },
                    {"Key": "data/file.txt", "Size": 512, "LastModified": None},
                ]
            }
        ]
        client.get_paginator.return_value = paginator
        return client

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_lists_all_files_with_wildcard(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        files = list(storage.list_files("data"))

        assert len(files) == 3
        assert all(isinstance(f, FileInfo) for f in files)
        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")

    def it_includes_file_metadata(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        files = list(storage.list_files("data"))

        assert files[0].key == "data/file1.csv"
        assert files[0].size_bytes == 1024
        assert files[0].last_modified is not None


class DescribeS3StorageGetSizeBytes:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_returns_object_size(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        mock_s3_client.head_object.return_value = {"ContentLength": 4096}

        size = storage.get_size_bytes("data/large.bin")

        assert size == 4096
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="data/large.bin"
        )

    def it_raises_error_when_object_does_not_exist(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        error = ClientError({"Error": {"Code": "404"}}, "head_object")
        mock_s3_client.head_object.side_effect = error

        with pytest.raises(FileDoesNotExistError):
            storage.get_size_bytes("missing.bin")


class DescribeS3StorageReadWriteBytes:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_reads_bytes_from_object(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        mock_body = MagicMock()
        mock_body.read.return_value = b"test data"
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        result = storage.read_bytes("data/file.bin")

        assert result == b"test data"
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="data/file.bin"
        )

    def it_writes_bytes_to_object(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        data = b"binary content"

        storage.write_bytes(data, "output/data.bin")

        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="output/data.bin", Body=data
        )

    def it_raises_error_reading_nonexistent_object(
        self, storage: S3Storage, mock_s3_client: MagicMock
    ) -> None:
        error = ClientError({"Error": {"Code": "NoSuchKey"}}, "get_object")
        mock_s3_client.get_object.side_effect = error

        with pytest.raises(FileDoesNotExistError):
            storage.read_bytes("missing.bin")


class DescribeS3StorageReadWriteParquet:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    def it_writes_dataframe_as_parquet(
        self, storage: S3Storage, mock_s3_client: MagicMock, sample_dataframe: pl.DataFrame
    ) -> None:
        storage.write_parquet(sample_dataframe, "data/output.parquet")

        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args.kwargs["Bucket"] == "test-bucket"
        assert call_args.kwargs["Key"] == "data/output.parquet"
        assert isinstance(call_args.kwargs["Body"], bytes)

    def it_reads_parquet_as_dataframe(
        self, storage: S3Storage, mock_s3_client: MagicMock, sample_dataframe: pl.DataFrame
    ) -> None:
        buffer = BytesIO()
        sample_dataframe.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        mock_body = MagicMock()
        mock_body.read.return_value = parquet_bytes
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        result = storage.read_parquet("data/input.parquet")

        assert result.equals(sample_dataframe)


class DescribeS3StorageSinkParquet:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    @pytest.fixture
    def sample_lazyframe(self) -> pl.LazyFrame:
        return pl.LazyFrame({"x": [10, 20, 30], "y": [100, 200, 300]})

    def it_sinks_lazyframe_to_s3(
        self, storage: S3Storage, mock_s3_client: MagicMock, sample_lazyframe: pl.LazyFrame
    ) -> None:
        storage.sink_parquet(sample_lazyframe, "data/lazy.parquet")

        mock_s3_client.upload_file.assert_called_once()
        call_args = mock_s3_client.upload_file.call_args
        assert call_args[0][1] == "test-bucket"
        assert call_args[0][2] == "data/lazy.parquet"


class DescribeS3StorageReadCsv:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_reads_csv_as_dataframe(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        csv_data = b"col1,col2\n1,a\n2,b"
        mock_body = MagicMock()
        mock_body.read.return_value = csv_data
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        df = storage.read_csv("data/file.csv")

        assert df.shape == (2, 2)
        assert df.columns == ["col1", "col2"]


class DescribeS3StorageScanParquet:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        # Create a real parquet file for scanning
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_path = tmp_path / "test.parquet"
        df.write_parquet(parquet_path)

        mock_s3_client.download_file.side_effect = lambda bucket, key, path: None

        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_downloads_and_scans_parquet_as_lazyframe(
        self, storage: S3Storage, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        # Pre-populate cache with the test file
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache_path = tmp_path / "data" / "scan.parquet"
        cache_path.parent.mkdir(parents=True)
        df.write_parquet(cache_path)

        lf = storage.scan_parquet("data/scan.parquet")

        assert isinstance(lf, pl.LazyFrame)


class DescribeS3StorageScanCsv:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_downloads_and_scans_csv_as_lazyframe(
        self, storage: S3Storage, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        # Pre-populate cache with CSV file
        csv_path = tmp_path / "data" / "scan.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("id,value\n1,10\n2,20")

        lf = storage.scan_csv("data/scan.csv")

        assert isinstance(lf, pl.LazyFrame)


class DescribeS3StorageReadWriteJson:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    @pytest.fixture
    def sample_json(self) -> dict[str, Any]:
        return {"key": "value", "number": 123, "list": [1, 2, 3]}

    def it_writes_json_to_object(
        self, storage: S3Storage, mock_s3_client: MagicMock, sample_json: dict[str, Any]
    ) -> None:
        storage.write_json(sample_json, "config/settings.json")

        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args.kwargs["Key"] == "config/settings.json"

    def it_reads_json_from_object(
        self, storage: S3Storage, mock_s3_client: MagicMock, sample_json: dict[str, Any]
    ) -> None:
        import json

        json_bytes = json.dumps(sample_json).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = json_bytes
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        result = storage.read_json("config/settings.json")

        assert result == sample_json


class DescribeS3StorageReadWriteJoblib:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_s3_client: MagicMock, tmp_path: Path) -> S3Storage:
        return S3Storage(mock_s3_client, "test-bucket", cache_dir=tmp_path)

    def it_writes_joblib_to_object(self, storage: S3Storage, mock_s3_client: MagicMock) -> None:
        obj = {"model": "test", "data": [1, 2, 3]}

        storage.write_joblib(obj, "models/model.joblib")

        mock_s3_client.upload_file.assert_called_once()
        call_args = mock_s3_client.upload_file.call_args
        assert call_args[0][1] == "test-bucket"
        assert call_args[0][2] == "models/model.joblib"

    def it_reads_joblib_from_cache(
        self, storage: S3Storage, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        import joblib

        obj = {"model": "test", "data": [1, 2, 3]}
        cache_path = tmp_path / "models" / "model.joblib"
        cache_path.parent.mkdir(parents=True)
        joblib.dump(obj, cache_path)

        result = storage.read_joblib("models/model.joblib")

        assert result == obj
