from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from google.api_core.exceptions import NotFound
import polars as pl
import pytest

from experiments.storage.errors import FileDoesNotExistError
from experiments.storage.gcs import GCSStorage
from experiments.storage.interface import FileInfo


class DescribeGCSStorageExists:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", prefix="data", cache_dir=tmp_path)

    def it_returns_true_when_blob_exists(
        self, storage: GCSStorage, mock_gcs_client: MagicMock
    ) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        result = storage.exists("file.txt")

        assert result is True

    def it_returns_false_when_blob_does_not_exist(
        self, storage: GCSStorage, mock_gcs_client: MagicMock
    ) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        result = storage.exists("missing.txt")

        assert result is False


class DescribeGCSStorageDelete:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_deletes_existing_blob(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        storage.delete("file.txt")

        mock_blob.delete.assert_called_once()

    def it_raises_error_when_blob_does_not_exist(
        self, storage: GCSStorage, mock_gcs_client: MagicMock
    ) -> None:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        with pytest.raises(FileDoesNotExistError) as exc_info:
            storage.delete("missing.txt")

        assert exc_info.value.uri == "missing.txt"


class DescribeGCSStorageListFiles:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        client = MagicMock()
        bucket = MagicMock()

        # Create mock blobs
        blob1 = MagicMock()
        blob1.name = "storage/data/file1.csv"
        blob1.size = 1024
        blob1.updated = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        blob2 = MagicMock()
        blob2.name = "storage/data/file2.csv"
        blob2.size = 2048
        blob2.updated = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        blob3 = MagicMock()
        blob3.name = "storage/data/file.txt"
        blob3.size = 512
        blob3.updated = None

        bucket.list_blobs.return_value = [blob1, blob2, blob3]
        client.bucket.return_value = bucket
        return client

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", prefix="storage", cache_dir=tmp_path)

    def it_lists_all_files_with_wildcard(
        self, storage: GCSStorage, mock_gcs_client: MagicMock
    ) -> None:
        files = list(storage.list_files("data"))

        assert len(files) == 3
        assert all(isinstance(f, FileInfo) for f in files)

    def it_includes_file_metadata(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        files = list(storage.list_files("data"))

        assert files[0].key == "data/file1.csv"
        assert files[0].size_bytes == 1024
        assert files[0].last_modified is not None


class DescribeGCSStorageGetSizeBytes:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_returns_blob_size(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        mock_blob = MagicMock()
        mock_blob.size = 4096
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        size = storage.get_size_bytes("large.bin")

        assert size == 4096
        mock_blob.reload.assert_called_once()

    def it_raises_error_when_blob_does_not_exist(
        self, storage: GCSStorage, mock_gcs_client: MagicMock
    ) -> None:
        mock_blob = MagicMock()
        mock_blob.reload.side_effect = NotFound("Not found")
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        with pytest.raises(FileDoesNotExistError):
            storage.get_size_bytes("missing.bin")


class DescribeGCSStorageReadWriteBytes:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_reads_bytes_from_blob(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = b"test data"
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        result = storage.read_bytes("file.bin")

        assert result == b"test data"
        mock_blob.download_as_bytes.assert_called_once()

    def it_writes_bytes_to_blob(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob
        data = b"binary content"

        storage.write_bytes(data, "output/data.bin")

        mock_blob.upload_from_string.assert_called_once_with(data)

    def it_raises_error_reading_nonexistent_blob(
        self, storage: GCSStorage, mock_gcs_client: MagicMock
    ) -> None:
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = NotFound("Not found")
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        with pytest.raises(FileDoesNotExistError):
            storage.read_bytes("missing.bin")


class DescribeGCSStorageReadWriteParquet:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    def it_writes_dataframe_as_parquet(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, sample_dataframe: pl.DataFrame
    ) -> None:
        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        storage.write_parquet(sample_dataframe, "data/output.parquet")

        mock_blob.upload_from_string.assert_called_once()
        call_args = mock_blob.upload_from_string.call_args
        assert isinstance(call_args[0][0], bytes)

    def it_reads_parquet_as_dataframe(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, sample_dataframe: pl.DataFrame
    ) -> None:
        buffer = BytesIO()
        sample_dataframe.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = parquet_bytes
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        result = storage.read_parquet("data/input.parquet")

        assert result.equals(sample_dataframe)


class DescribeGCSStorageSinkParquet:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    @pytest.fixture
    def sample_lazyframe(self) -> pl.LazyFrame:
        return pl.LazyFrame({"x": [10, 20, 30], "y": [100, 200, 300]})

    def it_sinks_lazyframe_to_gcs(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, sample_lazyframe: pl.LazyFrame
    ) -> None:
        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        storage.sink_parquet(sample_lazyframe, "data/lazy.parquet")

        mock_blob.upload_from_filename.assert_called_once()


class DescribeGCSStorageReadCsv:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_reads_csv_as_dataframe(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        csv_data = b"col1,col2\n1,a\n2,b"
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = csv_data
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        df = storage.read_csv("data/file.csv")

        assert df.shape == (2, 2)
        assert df.columns == ["col1", "col2"]


class DescribeGCSStorageScanParquet:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_downloads_and_scans_parquet_as_lazyframe(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, tmp_path: Path
    ) -> None:
        # Pre-populate cache with the test file
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache_path = tmp_path / "data" / "scan.parquet"
        cache_path.parent.mkdir(parents=True)
        df.write_parquet(cache_path)

        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        lf = storage.scan_parquet("data/scan.parquet")

        assert isinstance(lf, pl.LazyFrame)


class DescribeGCSStorageScanCsv:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_downloads_and_scans_csv_as_lazyframe(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, tmp_path: Path
    ) -> None:
        # Pre-populate cache with CSV file
        csv_path = tmp_path / "data" / "scan.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("id,value\n1,10\n2,20")

        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        lf = storage.scan_csv("data/scan.csv")

        assert isinstance(lf, pl.LazyFrame)


class DescribeGCSStorageReadWriteJson:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    @pytest.fixture
    def sample_json(self) -> dict[str, Any]:
        return {"key": "value", "number": 123, "list": [1, 2, 3]}

    def it_writes_json_to_blob(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, sample_json: dict[str, Any]
    ) -> None:
        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        storage.write_json(sample_json, "config/settings.json")

        mock_blob.upload_from_string.assert_called_once()

    def it_reads_json_from_blob(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, sample_json: dict[str, Any]
    ) -> None:
        import json

        json_bytes = json.dumps(sample_json).encode()
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = json_bytes
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        result = storage.read_json("config/settings.json")

        assert result == sample_json


class DescribeGCSStorageReadWriteJoblib:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def storage(self, mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorage:
        return GCSStorage(mock_gcs_client, "test-bucket", cache_dir=tmp_path)

    def it_writes_joblib_to_blob(self, storage: GCSStorage, mock_gcs_client: MagicMock) -> None:
        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob
        obj = {"model": "test", "data": [1, 2, 3]}

        storage.write_joblib(obj, "models/model.joblib")

        mock_blob.upload_from_filename.assert_called_once()

    def it_reads_joblib_from_cache(
        self, storage: GCSStorage, mock_gcs_client: MagicMock, tmp_path: Path
    ) -> None:
        import joblib

        obj = {"model": "test", "data": [1, 2, 3]}
        cache_path = tmp_path / "models" / "model.joblib"
        cache_path.parent.mkdir(parents=True)
        joblib.dump(obj, cache_path)

        mock_blob = MagicMock()
        mock_gcs_client.bucket.return_value.blob.return_value = mock_blob

        result = storage.read_joblib("models/model.joblib")

        assert result == obj


class DescribeGCSStoragePrefixHandling:
    @pytest.fixture
    def mock_gcs_client(self) -> MagicMock:
        return MagicMock()

    def it_constructs_blob_name_with_prefix(
        self, mock_gcs_client: MagicMock, tmp_path: Path
    ) -> None:
        storage = GCSStorage(mock_gcs_client, "bucket", prefix="data/storage", cache_dir=tmp_path)

        blob_name = storage._blob_name("file.txt")

        assert blob_name == "data/storage/file.txt"

    def it_constructs_blob_name_without_prefix(
        self, mock_gcs_client: MagicMock, tmp_path: Path
    ) -> None:
        storage = GCSStorage(mock_gcs_client, "bucket", prefix="", cache_dir=tmp_path)

        blob_name = storage._blob_name("file.txt")

        assert blob_name == "file.txt"
