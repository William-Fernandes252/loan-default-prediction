"""Tests for S3Storage class."""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import Mock

from botocore.exceptions import ClientError
import joblib
import polars as pl
import pytest

# Add parent directory to path to import storage_new modules directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.services.storage_new.errors import FileDoesNotExistError, StorageError
from experiments.services.storage_new.s3 import S3Storage


class DescribeS3Storage:
    """Tests for the S3Storage class."""

    @pytest.fixture
    def s3_client(self) -> Mock:
        """Create a mock S3 client."""
        return Mock()

    @pytest.fixture
    def storage(self, s3_client: Mock, tmp_path: Path) -> S3Storage:
        """Create an S3Storage instance with mock client."""
        return S3Storage(
            s3_client=s3_client,
            bucket_name="test-bucket",
            cache_dir=tmp_path / "cache",
        )

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample Polars DataFrame for testing."""
        return pl.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

    class DescribeExists:
        """Tests for the exists method."""

        def it_returns_true_for_existing_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify returns True when file exists."""
            s3_client.head_object.return_value = {"ContentLength": 100}

            assert storage.exists("test.txt") is True

            s3_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test.txt")

        def it_returns_false_for_nonexistent_file(
            self, storage: S3Storage, s3_client: Mock
        ) -> None:
            """Verify returns False when file doesn't exist."""
            error = ClientError({"Error": {"Code": "404"}}, "HeadObject")
            s3_client.head_object.side_effect = error

            assert storage.exists("nonexistent.txt") is False

        def it_raises_for_other_errors(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify raises StorageError for non-404 errors."""
            error = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
            s3_client.head_object.side_effect = error

            with pytest.raises(StorageError):
                storage.exists("test.txt")

    class DescribeDelete:
        """Tests for the delete method."""

        def it_deletes_existing_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify file is deleted."""
            s3_client.head_object.return_value = {"ContentLength": 100}

            storage.delete("test.txt")

            s3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test.txt")

        def it_raises_for_nonexistent_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            error = ClientError({"Error": {"Code": "404"}}, "HeadObject")
            s3_client.head_object.side_effect = error

            with pytest.raises(FileDoesNotExistError):
                storage.delete("nonexistent.txt")

    class DescribeListFiles:
        """Tests for the list_files method."""

        def it_lists_all_files_with_default_pattern(
            self, storage: S3Storage, s3_client: Mock
        ) -> None:
            """Verify lists all files with default pattern."""
            paginator = Mock()
            s3_client.get_paginator.return_value = paginator
            paginator.paginate.return_value = [
                {
                    "Contents": [
                        {
                            "Key": "file1.txt",
                            "Size": 100,
                            "LastModified": datetime(2025, 1, 1, tzinfo=timezone.utc),
                        },
                        {
                            "Key": "file2.txt",
                            "Size": 200,
                            "LastModified": datetime(2025, 1, 2, tzinfo=timezone.utc),
                        },
                    ]
                }
            ]

            files = list(storage.list_files("prefix/"))

            assert len(files) == 2
            assert files[0].key == "file1.txt"
            assert files[0].size_bytes == 100
            assert files[1].key == "file2.txt"

        def it_filters_by_pattern(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify filters files by glob pattern."""
            paginator = Mock()
            s3_client.get_paginator.return_value = paginator
            paginator.paginate.return_value = [
                {
                    "Contents": [
                        {"Key": "file1.txt", "Size": 100},
                        {"Key": "file2.csv", "Size": 200},
                        {"Key": "file3.txt", "Size": 300},
                    ]
                }
            ]

            files = list(storage.list_files("", "*.txt"))

            assert len(files) == 2
            keys = {f.key for f in files}
            assert "file1.txt" in keys
            assert "file3.txt" in keys

        def it_handles_empty_results(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify handles pages without Contents."""
            paginator = Mock()
            s3_client.get_paginator.return_value = paginator
            paginator.paginate.return_value = [{}]

            files = list(storage.list_files("empty/"))

            assert files == []

    class DescribeGetSizeBytes:
        """Tests for the get_size_bytes method."""

        def it_returns_file_size(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify returns correct file size in bytes."""
            s3_client.head_object.return_value = {"ContentLength": 1024}

            size = storage.get_size_bytes("test.txt")

            assert size == 1024

        def it_raises_for_nonexistent_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            error = ClientError({"Error": {"Code": "404"}}, "HeadObject")
            s3_client.head_object.side_effect = error

            with pytest.raises(FileDoesNotExistError):
                storage.get_size_bytes("nonexistent.txt")

    class DescribeReadBytes:
        """Tests for the read_bytes method."""

        def it_reads_file_contents(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify reads file contents as bytes."""
            body = Mock()
            body.read.return_value = b"test content"
            s3_client.get_object.return_value = {"Body": body}

            content = storage.read_bytes("test.txt")

            assert content == b"test content"
            s3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test.txt")

        def it_raises_for_nonexistent_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            error = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            s3_client.get_object.side_effect = error

            with pytest.raises(FileDoesNotExistError):
                storage.read_bytes("nonexistent.txt")

    class DescribeWriteBytes:
        """Tests for the write_bytes method."""

        def it_writes_bytes_to_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify writes bytes to file."""
            data = b"test content"

            storage.write_bytes(data, "test.txt")

            s3_client.put_object.assert_called_once_with(
                Bucket="test-bucket", Key="test.txt", Body=data
            )

    class DescribeReadParquet:
        """Tests for the read_parquet method."""

        def it_reads_parquet_file(
            self,
            storage: S3Storage,
            s3_client: Mock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads parquet file into DataFrame."""
            buffer = BytesIO()
            sample_dataframe.write_parquet(buffer)
            buffer.seek(0)

            body = Mock()
            body.read.return_value = buffer.read()
            s3_client.get_object.return_value = {"Body": body}

            df = storage.read_parquet("data.parquet")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

    class DescribeWriteParquet:
        """Tests for the write_parquet method."""

        def it_writes_dataframe_to_parquet(
            self,
            storage: S3Storage,
            s3_client: Mock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to parquet file."""
            storage.write_parquet(sample_dataframe, "data.parquet")

            s3_client.put_object.assert_called_once()
            call_args = s3_client.put_object.call_args
            assert call_args[1]["Bucket"] == "test-bucket"
            assert call_args[1]["Key"] == "data.parquet"
            assert isinstance(call_args[1]["Body"], bytes)

    class DescribeSinkParquet:
        """Tests for the sink_parquet method."""

        def it_sinks_lazyframe_to_parquet(
            self,
            storage: S3Storage,
            s3_client: Mock,
            sample_dataframe: pl.DataFrame,
            tmp_path: Path,
        ) -> None:
            """Verify sinks LazyFrame to parquet file."""
            lf = sample_dataframe.lazy()

            storage.sink_parquet(lf, "data.parquet")

            s3_client.upload_file.assert_called_once()
            call_args = s3_client.upload_file.call_args[0]
            assert call_args[1] == "test-bucket"
            assert call_args[2] == "data.parquet"

    class DescribeReadCsv:
        """Tests for the read_csv method."""

        def it_reads_csv_file(
            self,
            storage: S3Storage,
            s3_client: Mock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads CSV file into DataFrame."""
            buffer = BytesIO()
            sample_dataframe.write_csv(buffer)
            buffer.seek(0)

            body = Mock()
            body.read.return_value = buffer.read()
            s3_client.get_object.return_value = {"Body": body}

            df = storage.read_csv("data.csv")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

    class DescribeScanParquet:
        """Tests for the scan_parquet method."""

        def it_scans_parquet_file(
            self,
            storage: S3Storage,
            s3_client: Mock,
            sample_dataframe: pl.DataFrame,
            tmp_path: Path,
        ) -> None:
            """Verify scans parquet file into LazyFrame."""
            # Create a temporary parquet file for download_file to work with
            parquet_path = tmp_path / "cache" / "data.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            sample_dataframe.write_parquet(parquet_path)

            lf = storage.scan_parquet("data.parquet")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape

        def it_raises_for_nonexistent_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            error = ClientError({"Error": {"Code": "404"}}, "DownloadFile")
            s3_client.download_file.side_effect = error

            with pytest.raises(FileDoesNotExistError):
                storage.scan_parquet("nonexistent.parquet")

    class DescribeScanCsv:
        """Tests for the scan_csv method."""

        def it_scans_csv_file(
            self,
            storage: S3Storage,
            s3_client: Mock,
            sample_dataframe: pl.DataFrame,
            tmp_path: Path,
        ) -> None:
            """Verify scans CSV file into LazyFrame."""
            # Create a temporary CSV file for download_file to work with
            csv_path = tmp_path / "cache" / "data.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            sample_dataframe.write_csv(csv_path)

            lf = storage.scan_csv("data.csv")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape

    class DescribeReadJson:
        """Tests for the read_json method."""

        def it_reads_json_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify reads JSON file into dictionary."""
            data = {"key": "value", "number": 42}
            body = Mock()
            body.read.return_value = json.dumps(data).encode("utf-8")
            s3_client.get_object.return_value = {"Body": body}

            result = storage.read_json("data.json")

            assert result == data

    class DescribeWriteJson:
        """Tests for the write_json method."""

        def it_writes_json_to_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify writes dictionary to JSON file."""
            data = {"key": "value", "number": 42}

            storage.write_json(data, "data.json")

            s3_client.put_object.assert_called_once()
            call_args = s3_client.put_object.call_args
            assert call_args[1]["Key"] == "data.json"
            body = call_args[1]["Body"]
            assert json.loads(body.decode("utf-8")) == data

    class DescribeReadJoblib:
        """Tests for the read_joblib method."""

        def it_reads_joblib_file(
            self,
            storage: S3Storage,
            s3_client: Mock,
            tmp_path: Path,
        ) -> None:
            """Verify reads joblib-serialized object."""
            data = {"key": "value", "list": [1, 2, 3]}
            joblib_path = tmp_path / "cache" / "data.joblib"
            joblib_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(data, joblib_path)

            result = storage.read_joblib("data.joblib")

            assert result == data

        def it_raises_for_nonexistent_file(self, storage: S3Storage, s3_client: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            error = ClientError({"Error": {"Code": "404"}}, "DownloadFile")
            s3_client.download_file.side_effect = error

            with pytest.raises(FileDoesNotExistError):
                storage.read_joblib("nonexistent.joblib")

    class DescribeWriteJoblib:
        """Tests for the write_joblib method."""

        def it_writes_joblib_to_file(
            self,
            storage: S3Storage,
            s3_client: Mock,
        ) -> None:
            """Verify writes object to joblib file."""
            data = {"key": "value", "list": [1, 2, 3]}

            storage.write_joblib(data, "data.joblib")

            s3_client.upload_file.assert_called_once()
            call_args = s3_client.upload_file.call_args[0]
            assert call_args[1] == "test-bucket"
            assert call_args[2] == "data.joblib"
