"""Tests for experiments.services.storage_gcs module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from experiments.services.storage import FileDoesNotExistError, StorageError
from experiments.services.storage.gcs import GCSStorageService, create_gcs_client


@pytest.fixture
def mock_gcs_client() -> MagicMock:
    """Create a mock GCS client."""
    return MagicMock()


@pytest.fixture
def storage(mock_gcs_client: MagicMock, tmp_path: Path) -> GCSStorageService:
    """Create a GCSStorageService with a mocked client."""
    return GCSStorageService(
        client=mock_gcs_client,
        bucket="test-bucket",
        prefix="experiments",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def storage_no_bucket(mock_gcs_client: MagicMock) -> GCSStorageService:
    """Create a GCSStorageService without a default bucket."""
    return GCSStorageService(client=mock_gcs_client)


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )


class DescribeGCSStorageService:
    """Tests for the GCSStorageService class."""

    class DescribeInit:
        """Tests for initialization."""

        def it_initializes_with_client(self, mock_gcs_client: MagicMock) -> None:
            """Verify initializes with injected client."""
            storage = GCSStorageService(client=mock_gcs_client)

            assert storage._client is mock_gcs_client

        def it_initializes_with_bucket_and_prefix(self, mock_gcs_client: MagicMock) -> None:
            """Verify initializes with bucket and prefix."""
            storage = GCSStorageService(
                client=mock_gcs_client,
                bucket="my-bucket",
                prefix="data/",
            )

            assert storage._bucket_name == "my-bucket"
            assert storage._prefix == "data"  # Trailing slash stripped

        def it_initializes_with_cache_dir(
            self, mock_gcs_client: MagicMock, tmp_path: Path
        ) -> None:
            """Verify initializes with custom cache directory."""
            cache_dir = tmp_path / "cache"

            storage = GCSStorageService(
                client=mock_gcs_client,
                cache_dir=cache_dir,
            )

            assert storage._cache_dir == cache_dir

    class DescribeParseGcsPath:
        """Tests for the _parse_gcs_path method."""

        def it_parses_gs_uri(self, storage: GCSStorageService) -> None:
            """Verify parses gs:// URIs correctly."""
            bucket, blob = storage._parse_gcs_path("gs://my-bucket/path/to/file.txt")

            assert bucket == "my-bucket"
            assert blob == "path/to/file.txt"

        def it_raises_for_file_uri(self, storage: GCSStorageService) -> None:
            """Verify raises StorageError for file:// URIs."""
            with pytest.raises(StorageError) as exc_info:
                storage._parse_gcs_path("file:///tmp/test.txt")

            assert "does not support file://" in str(exc_info.value)

    class DescribeExists:
        """Tests for the exists method."""

        def it_returns_true_when_blob_exists(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify returns True when blob exists."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            result = storage.exists("gs://bucket/file.txt")

            assert result is True

        def it_returns_false_when_blob_not_exists(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify returns False when blob doesn't exist."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = False
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            result = storage.exists("gs://bucket/file.txt")

            assert result is False

    class DescribeDelete:
        """Tests for the delete method."""

        def it_deletes_single_blob(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify deletes a single blob."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_bucket.blob.return_value = mock_blob
            mock_bucket.list_blobs.return_value = []
            mock_gcs_client.bucket.return_value = mock_bucket

            storage.delete("gs://bucket/file.txt")

            mock_blob.delete.assert_called_once()

        def it_deletes_prefix_with_blobs(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify deletes all blobs under a prefix."""
            mock_bucket = MagicMock()
            mock_blob1 = MagicMock()
            mock_blob2 = MagicMock()
            # First call for checking if it's a prefix
            mock_bucket.list_blobs.side_effect = [
                [mock_blob1],  # Has content under prefix
                [mock_blob1, mock_blob2],  # All blobs to delete
            ]
            mock_gcs_client.bucket.return_value = mock_bucket

            storage.delete("gs://bucket/path")

            mock_blob1.delete.assert_called()
            mock_blob2.delete.assert_called()

    class DescribeListFiles:
        """Tests for the list_files method."""

        def it_lists_blobs_in_prefix(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify lists blobs in GCS prefix."""
            mock_bucket = MagicMock()
            mock_blob1 = MagicMock()
            mock_blob1.name = "data/file1.parquet"
            mock_blob2 = MagicMock()
            mock_blob2.name = "data/file2.parquet"
            mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
            mock_gcs_client.bucket.return_value = mock_bucket

            files = storage.list_files("gs://bucket/data", "*.parquet")

            assert len(files) == 2
            assert "gs://bucket/data/file1.parquet" in files
            assert "gs://bucket/data/file2.parquet" in files

        def it_returns_empty_list_on_error(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify returns empty list on error."""
            mock_gcs_client.bucket.side_effect = Exception("Network error")

            files = storage.list_files("gs://bucket/data")

            assert files == []

        def it_skips_directory_markers(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify skips blobs ending with /."""
            mock_bucket = MagicMock()
            mock_blob1 = MagicMock()
            mock_blob1.name = "data/"  # Directory marker
            mock_blob2 = MagicMock()
            mock_blob2.name = "data/file.txt"
            mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
            mock_gcs_client.bucket.return_value = mock_bucket

            files = storage.list_files("gs://bucket/data")

            assert len(files) == 1
            assert "file.txt" in files[0]

    class DescribeMakedirs:
        """Tests for the makedirs method."""

        def it_does_nothing_for_gcs(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify makedirs is a no-op for GCS (directories are implicit)."""
            storage.makedirs("gs://bucket/path/to/dir")

            # Should not call any GCS methods for creating objects
            assert not mock_gcs_client.bucket.called

    class DescribeGetSizeBytes:
        """Tests for the get_size_bytes method."""

        def it_returns_blob_size(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify returns blob size from metadata."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.size = 12345
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            size = storage.get_size_bytes("gs://bucket/file.txt")

            assert size == 12345
            mock_blob.reload.assert_called_once()

        def it_raises_for_nonexistent_blob(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify raises FileDoesNotExistError for missing blob."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.size = None  # Indicates blob doesn't exist
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with pytest.raises(FileDoesNotExistError):
                storage.get_size_bytes("gs://bucket/missing.txt")

    class DescribeReadBytes:
        """Tests for the read_bytes method."""

        def it_reads_blob_content(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify reads blob content as bytes."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_blob.download_as_bytes.return_value = b"content"
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            content = storage.read_bytes("gs://bucket/file.txt")

            assert content == b"content"

        def it_raises_for_nonexistent_blob(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify raises FileDoesNotExistError for missing blob."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = False
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with pytest.raises(FileDoesNotExistError):
                storage.read_bytes("gs://bucket/missing.txt")

    class DescribeWriteBytes:
        """Tests for the write_bytes method."""

        def it_writes_bytes_to_blob(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify writes bytes using upload_from_string."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket
            data = b"content"

            storage.write_bytes(data, "gs://bucket/file.txt")

            mock_blob.upload_from_string.assert_called_once_with(data)

        def it_raises_on_error(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify raises StorageError on write failure."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.upload_from_string.side_effect = Exception("Write failed")
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with pytest.raises(StorageError):
                storage.write_bytes(b"data", "gs://bucket/file.txt")

    class DescribeOpenBinary:
        """Tests for the open_binary context manager."""

        def it_opens_for_reading(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify opens blob for reading."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_blob.download_as_bytes.return_value = b"content"
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with storage.open_binary("gs://bucket/file.txt", "rb") as f:
                content = f.read()

            assert content == b"content"

        def it_opens_for_writing(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify opens buffer for writing and uploads on close."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with storage.open_binary("gs://bucket/file.txt", "wb") as f:
                f.write(b"content")

            mock_blob.upload_from_file.assert_called_once()

        def it_raises_for_unsupported_mode(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify raises for unsupported mode."""
            with pytest.raises(StorageError) as exc_info:
                with storage.open_binary("gs://bucket/file.txt", "a"):
                    pass

            assert "Unsupported mode" in str(exc_info.value)

    class DescribeReadParquet:
        """Tests for the read_parquet method."""

        def it_reads_parquet_from_gcs(
            self,
            storage: GCSStorageService,
            mock_gcs_client: MagicMock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads parquet file from GCS."""
            # Create parquet bytes
            buffer = io.BytesIO()
            sample_dataframe.write_parquet(buffer)
            buffer.seek(0)

            # Mock download_to_filename to write the parquet data to the temp file
            def mock_download_to_filename(filename):
                with open(filename, "wb") as f:
                    f.write(buffer.getvalue())

            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_blob.download_to_filename.side_effect = mock_download_to_filename
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            df = storage.read_parquet("gs://bucket/data.parquet")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

    class DescribeWriteParquet:
        """Tests for the write_parquet method."""

        def it_writes_parquet_to_gcs(
            self,
            storage: GCSStorageService,
            mock_gcs_client: MagicMock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to GCS as parquet."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            storage.write_parquet(sample_dataframe, "gs://bucket/data.parquet")

            mock_blob.upload_from_file.assert_called_once()

    class DescribeReadJson:
        """Tests for the read_json method."""

        def it_reads_json_from_gcs(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify reads JSON from GCS."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_blob.download_as_bytes.return_value = b'{"key": "value"}'
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            data = storage.read_json("gs://bucket/data.json")

            assert data == {"key": "value"}

    class DescribeWriteJson:
        """Tests for the write_json method."""

        def it_writes_json_to_gcs(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock
        ) -> None:
            """Verify writes JSON to GCS."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket
            data = {"key": "value"}

            storage.write_json(data, "gs://bucket/data.json")

            mock_blob.upload_from_string.assert_called_once()

    class DescribeLocalCache:
        """Tests for the local_cache context manager."""

        def it_downloads_blob_to_cache(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock, tmp_path: Path
        ) -> None:
            """Verify downloads blob to local cache."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with storage.local_cache("gs://bucket/file.txt") as local_path:
                assert isinstance(local_path, Path)
                mock_blob.download_to_filename.assert_called_once()

        def it_cleans_up_cached_file(
            self, storage: GCSStorageService, mock_gcs_client: MagicMock, tmp_path: Path
        ) -> None:
            """Verify cleans up cached file after context exits."""
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            # Create a file that would be cleaned up
            mock_blob.download_to_filename.side_effect = lambda path: Path(path).write_text(
                "content"
            )
            mock_bucket.blob.return_value = mock_blob
            mock_gcs_client.bucket.return_value = mock_bucket

            with storage.local_cache("gs://bucket/file.txt") as local_path:
                cached_path = local_path

            assert not cached_path.exists()


class DescribeCreateGcsClient:
    """Tests for the create_gcs_client factory function."""

    @patch("google.cloud.storage.Client")
    def it_creates_default_client(self, mock_client_class: MagicMock) -> None:
        """Verify creates client with default credentials."""
        create_gcs_client()

        mock_client_class.assert_called_once_with()

    @patch("google.cloud.storage.Client")
    def it_creates_client_with_project(self, mock_client_class: MagicMock) -> None:
        """Verify creates client with explicit project."""
        create_gcs_client(project="my-project")

        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["project"] == "my-project"

    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    @patch("google.cloud.storage.Client")
    def it_creates_client_with_credentials_file(
        self, mock_client_class: MagicMock, mock_from_file: MagicMock
    ) -> None:
        """Verify creates client with service account credentials."""
        mock_creds = MagicMock()
        mock_from_file.return_value = mock_creds

        create_gcs_client(credentials_file="/path/to/creds.json")

        mock_from_file.assert_called_once_with("/path/to/creds.json")
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["credentials"] is mock_creds
