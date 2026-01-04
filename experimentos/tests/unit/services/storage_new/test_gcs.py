"""Tests for GCSStorage class."""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import Mock

from google.api_core.exceptions import NotFound
import joblib
import polars as pl
import pytest

# Add parent directory to path to import storage_new modules directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.services.storage_new.errors import FileDoesNotExistError
from experiments.services.storage_new.gcs import GCSStorage


class DescribeGCSStorage:
    """Tests for the GCSStorage class."""

    @pytest.fixture
    def gcs_client(self) -> Mock:
        """Create a mock GCS client."""
        return Mock()

    @pytest.fixture
    def bucket(self) -> Mock:
        """Create a mock GCS bucket."""
        return Mock()

    @pytest.fixture
    def storage(self, gcs_client: Mock, bucket: Mock, tmp_path: Path) -> GCSStorage:
        """Create a GCSStorage instance with mock client."""
        gcs_client.bucket.return_value = bucket
        return GCSStorage(
            gcs_client=gcs_client,
            bucket_name="test-bucket",
            prefix="data/storage",
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

    class DescribeBlobName:
        """Tests for the _blob_name helper method."""

        def it_constructs_blob_name_with_prefix(self, storage: GCSStorage) -> None:
            """Verify constructs full blob name with prefix."""
            result = storage._blob_name("test.txt")

            assert result == "data/storage/test.txt"

        def it_handles_nested_keys(self, storage: GCSStorage) -> None:
            """Verify handles nested paths correctly."""
            result = storage._blob_name("subdir/file.txt")

            assert result == "data/storage/subdir/file.txt"

    class DescribeExists:
        """Tests for the exists method."""

        def it_returns_true_for_existing_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify returns True when file exists."""
            blob = Mock()
            blob.exists.return_value = True
            bucket.blob.return_value = blob

            assert storage.exists("test.txt") is True

            bucket.blob.assert_called_once_with("data/storage/test.txt")

        def it_returns_false_for_nonexistent_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify returns False when file doesn't exist."""
            blob = Mock()
            blob.exists.return_value = False
            bucket.blob.return_value = blob

            assert storage.exists("nonexistent.txt") is False

    class DescribeDelete:
        """Tests for the delete method."""

        def it_deletes_existing_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify file is deleted."""
            blob = Mock()
            blob.exists.return_value = True
            bucket.blob.return_value = blob

            storage.delete("test.txt")

            blob.delete.assert_called_once()

        def it_raises_for_nonexistent_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            blob = Mock()
            blob.exists.return_value = False
            bucket.blob.return_value = blob

            with pytest.raises(FileDoesNotExistError):
                storage.delete("nonexistent.txt")

    class DescribeListFiles:
        """Tests for the list_files method."""

        def it_lists_all_files_with_default_pattern(
            self, storage: GCSStorage, bucket: Mock
        ) -> None:
            """Verify lists all files with default pattern."""
            blob1 = Mock()
            blob1.name = "data/storage/file1.txt"
            blob1.size = 100
            blob1.updated = datetime(2025, 1, 1, tzinfo=timezone.utc)

            blob2 = Mock()
            blob2.name = "data/storage/file2.txt"
            blob2.size = 200
            blob2.updated = datetime(2025, 1, 2, tzinfo=timezone.utc)

            bucket.list_blobs.return_value = [blob1, blob2]

            files = list(storage.list_files(""))

            assert len(files) == 2
            assert files[0].key == "file1.txt"
            assert files[0].size_bytes == 100
            assert files[1].key == "file2.txt"
            assert files[1].size_bytes == 200

        def it_filters_by_pattern(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify filters files by glob pattern."""
            blob1 = Mock()
            blob1.name = "data/storage/file1.txt"
            blob1.size = 100
            blob1.updated = None

            blob2 = Mock()
            blob2.name = "data/storage/file2.csv"
            blob2.size = 200
            blob2.updated = None

            blob3 = Mock()
            blob3.name = "data/storage/file3.txt"
            blob3.size = 300
            blob3.updated = None

            bucket.list_blobs.return_value = [blob1, blob2, blob3]

            files = list(storage.list_files("", "*.txt"))

            assert len(files) == 2
            keys = {f.key for f in files}
            assert "file1.txt" in keys
            assert "file3.txt" in keys

        def it_handles_prefix_correctly(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify uses full prefix with base prefix."""
            blob = Mock()
            blob.name = "data/storage/subdir/file.txt"
            blob.size = 100
            blob.updated = None

            bucket.list_blobs.return_value = [blob]

            list(storage.list_files("subdir"))

            bucket.list_blobs.assert_called_once_with(prefix="data/storage/subdir")

    class DescribeGetSizeBytes:
        """Tests for the get_size_bytes method."""

        def it_returns_file_size(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify returns correct file size in bytes."""
            blob = Mock()
            blob.size = 1024
            bucket.blob.return_value = blob

            size = storage.get_size_bytes("test.txt")

            assert size == 1024
            blob.reload.assert_called_once()

        def it_raises_for_nonexistent_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            blob = Mock()
            blob.reload.side_effect = NotFound("Not found")
            bucket.blob.return_value = blob

            with pytest.raises(FileDoesNotExistError):
                storage.get_size_bytes("nonexistent.txt")

    class DescribeReadBytes:
        """Tests for the read_bytes method."""

        def it_reads_file_contents(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify reads file contents as bytes."""
            blob = Mock()
            blob.download_as_bytes.return_value = b"test content"
            bucket.blob.return_value = blob

            content = storage.read_bytes("test.txt")

            assert content == b"test content"
            blob.download_as_bytes.assert_called_once()

        def it_raises_for_nonexistent_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            blob = Mock()
            blob.download_as_bytes.side_effect = NotFound("Not found")
            bucket.blob.return_value = blob

            with pytest.raises(FileDoesNotExistError):
                storage.read_bytes("nonexistent.txt")

    class DescribeWriteBytes:
        """Tests for the write_bytes method."""

        def it_writes_bytes_to_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify writes bytes to file."""
            blob = Mock()
            bucket.blob.return_value = blob
            data = b"test content"

            storage.write_bytes(data, "test.txt")

            blob.upload_from_string.assert_called_once_with(data)

    class DescribeReadParquet:
        """Tests for the read_parquet method."""

        def it_reads_parquet_file(
            self,
            storage: GCSStorage,
            bucket: Mock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads parquet file into DataFrame."""
            buffer = BytesIO()
            sample_dataframe.write_parquet(buffer)
            buffer.seek(0)

            blob = Mock()
            blob.download_as_bytes.return_value = buffer.read()
            bucket.blob.return_value = blob

            df = storage.read_parquet("data.parquet")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

    class DescribeWriteParquet:
        """Tests for the write_parquet method."""

        def it_writes_dataframe_to_parquet(
            self,
            storage: GCSStorage,
            bucket: Mock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to parquet file."""
            blob = Mock()
            bucket.blob.return_value = blob

            storage.write_parquet(sample_dataframe, "data.parquet")

            blob.upload_from_string.assert_called_once()
            call_args = blob.upload_from_string.call_args[0]
            assert isinstance(call_args[0], bytes)

    class DescribeSinkParquet:
        """Tests for the sink_parquet method."""

        def it_sinks_lazyframe_to_parquet(
            self,
            storage: GCSStorage,
            bucket: Mock,
            sample_dataframe: pl.DataFrame,
            tmp_path: Path,
        ) -> None:
            """Verify sinks LazyFrame to parquet file."""
            blob = Mock()
            bucket.blob.return_value = blob
            lf = sample_dataframe.lazy()

            storage.sink_parquet(lf, "data.parquet")

            blob.upload_from_filename.assert_called_once()

    class DescribeReadCsv:
        """Tests for the read_csv method."""

        def it_reads_csv_file(
            self,
            storage: GCSStorage,
            bucket: Mock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads CSV file into DataFrame."""
            buffer = BytesIO()
            sample_dataframe.write_csv(buffer)
            buffer.seek(0)

            blob = Mock()
            blob.download_as_bytes.return_value = buffer.read()
            bucket.blob.return_value = blob

            df = storage.read_csv("data.csv")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

    class DescribeScanParquet:
        """Tests for the scan_parquet method."""

        def it_scans_parquet_file(
            self,
            storage: GCSStorage,
            bucket: Mock,
            sample_dataframe: pl.DataFrame,
            tmp_path: Path,
        ) -> None:
            """Verify scans parquet file into LazyFrame."""
            # Create a temporary parquet file for download to work with
            parquet_path = tmp_path / "cache" / "data.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            sample_dataframe.write_parquet(parquet_path)

            blob = Mock()
            bucket.blob.return_value = blob

            lf = storage.scan_parquet("data.parquet")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape

        def it_raises_for_nonexistent_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            blob = Mock()
            blob.download_to_filename.side_effect = NotFound("Not found")
            bucket.blob.return_value = blob

            with pytest.raises(FileDoesNotExistError):
                storage.scan_parquet("nonexistent.parquet")

    class DescribeScanCsv:
        """Tests for the scan_csv method."""

        def it_scans_csv_file(
            self,
            storage: GCSStorage,
            bucket: Mock,
            sample_dataframe: pl.DataFrame,
            tmp_path: Path,
        ) -> None:
            """Verify scans CSV file into LazyFrame."""
            # Create a temporary CSV file for download to work with
            csv_path = tmp_path / "cache" / "data.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            sample_dataframe.write_csv(csv_path)

            blob = Mock()
            bucket.blob.return_value = blob

            lf = storage.scan_csv("data.csv")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape

    class DescribeReadJson:
        """Tests for the read_json method."""

        def it_reads_json_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify reads JSON file into dictionary."""
            data = {"key": "value", "number": 42}
            blob = Mock()
            blob.download_as_bytes.return_value = json.dumps(data).encode("utf-8")
            bucket.blob.return_value = blob

            result = storage.read_json("data.json")

            assert result == data

    class DescribeWriteJson:
        """Tests for the write_json method."""

        def it_writes_json_to_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify writes dictionary to JSON file."""
            blob = Mock()
            bucket.blob.return_value = blob
            data = {"key": "value", "number": 42}

            storage.write_json(data, "data.json")

            blob.upload_from_string.assert_called_once()
            call_args = blob.upload_from_string.call_args[0]
            assert json.loads(call_args[0].decode("utf-8")) == data

    class DescribeReadJoblib:
        """Tests for the read_joblib method."""

        def it_reads_joblib_file(
            self,
            storage: GCSStorage,
            bucket: Mock,
            tmp_path: Path,
        ) -> None:
            """Verify reads joblib-serialized object."""
            data = {"key": "value", "list": [1, 2, 3]}
            joblib_path = tmp_path / "cache" / "data.joblib"
            joblib_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(data, joblib_path)

            blob = Mock()
            bucket.blob.return_value = blob

            result = storage.read_joblib("data.joblib")

            assert result == data

        def it_raises_for_nonexistent_file(self, storage: GCSStorage, bucket: Mock) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            blob = Mock()
            blob.download_to_filename.side_effect = NotFound("Not found")
            bucket.blob.return_value = blob

            with pytest.raises(FileDoesNotExistError):
                storage.read_joblib("nonexistent.joblib")

    class DescribeWriteJoblib:
        """Tests for the write_joblib method."""

        def it_writes_joblib_to_file(
            self,
            storage: GCSStorage,
            bucket: Mock,
        ) -> None:
            """Verify writes object to joblib file."""
            blob = Mock()
            bucket.blob.return_value = blob
            data = {"key": "value", "list": [1, 2, 3]}

            storage.write_joblib(data, "data.joblib")

            blob.upload_from_filename.assert_called_once()
