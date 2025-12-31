"""Tests for experiments.services.storage_s3 module."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from experiments.services.storage import FileDoesNotExistError, StorageError
from experiments.services.storage.s3 import S3StorageService, create_s3_client


@pytest.fixture
def mock_s3_client() -> MagicMock:
    """Create a mock boto3 S3 client."""
    return MagicMock()


@pytest.fixture
def storage(mock_s3_client: MagicMock, tmp_path: Path) -> S3StorageService:
    """Create an S3StorageService with a mocked client."""
    return S3StorageService(
        client=mock_s3_client,
        bucket="test-bucket",
        prefix="experiments",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def storage_no_bucket(mock_s3_client: MagicMock) -> S3StorageService:
    """Create an S3StorageService without a default bucket."""
    return S3StorageService(client=mock_s3_client)


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


class DescribeS3StorageService:
    """Tests for the S3StorageService class."""

    class DescribeInit:
        """Tests for initialization."""

        def it_initializes_with_client(self, mock_s3_client: MagicMock) -> None:
            """Verify initializes with injected client."""
            storage = S3StorageService(client=mock_s3_client)

            assert storage._client is mock_s3_client

        def it_initializes_with_bucket_and_prefix(self, mock_s3_client: MagicMock) -> None:
            """Verify initializes with bucket and prefix."""
            storage = S3StorageService(
                client=mock_s3_client,
                bucket="my-bucket",
                prefix="data/",
            )

            assert storage._bucket == "my-bucket"
            assert storage._prefix == "data"  # Trailing slash stripped

        def it_initializes_with_cache_dir(self, mock_s3_client: MagicMock, tmp_path: Path) -> None:
            """Verify initializes with custom cache directory."""
            cache_dir = tmp_path / "cache"

            storage = S3StorageService(
                client=mock_s3_client,
                cache_dir=cache_dir,
            )

            assert storage._cache_dir == cache_dir

    class DescribeParseS3Path:
        """Tests for the _parse_s3_path method."""

        def it_parses_s3_uri(self, storage: S3StorageService) -> None:
            """Verify parses s3:// URIs correctly."""
            bucket, key = storage._parse_s3_path("s3://my-bucket/path/to/file.txt")

            assert bucket == "my-bucket"
            assert key == "path/to/file.txt"

        def it_raises_for_file_uri(self, storage: S3StorageService) -> None:
            """Verify raises StorageError for file:// URIs."""
            with pytest.raises(StorageError) as exc_info:
                storage._parse_s3_path("file:///tmp/test.txt")

            assert "does not support file://" in str(exc_info.value)

    class DescribeExists:
        """Tests for the exists method."""

        def it_returns_true_when_object_exists(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify returns True when object exists."""
            mock_s3_client.head_object.return_value = {"ContentLength": 100}

            result = storage.exists("s3://bucket/file.txt")

            assert result is True
            mock_s3_client.head_object.assert_called_once_with(Bucket="bucket", Key="file.txt")

        def it_returns_false_when_object_not_exists(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify returns False when object doesn't exist."""
            from botocore.exceptions import ClientError

            mock_s3_client.head_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "HeadObject"
            )

            result = storage.exists("s3://bucket/file.txt")

            assert result is False

    class DescribeDelete:
        """Tests for the delete method."""

        def it_deletes_single_object(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify deletes a single object."""
            mock_s3_client.list_objects_v2.return_value = {}

            storage.delete("s3://bucket/file.txt")

            mock_s3_client.delete_object.assert_called_once_with(Bucket="bucket", Key="file.txt")

        def it_deletes_prefix_with_objects(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify deletes all objects under a prefix."""
            mock_s3_client.list_objects_v2.return_value = {"Contents": [{"Key": "path/file1.txt"}]}
            mock_paginator = MagicMock()
            mock_paginator.paginate.return_value = [
                {"Contents": [{"Key": "path/file1.txt"}, {"Key": "path/file2.txt"}]}
            ]
            mock_s3_client.get_paginator.return_value = mock_paginator

            storage.delete("s3://bucket/path")

            mock_s3_client.delete_objects.assert_called_once()

    class DescribeListFiles:
        """Tests for the list_files method."""

        def it_lists_files_in_prefix(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify lists files in S3 prefix."""
            mock_paginator = MagicMock()
            mock_paginator.paginate.return_value = [
                {
                    "Contents": [
                        {"Key": "data/file1.parquet"},
                        {"Key": "data/file2.parquet"},
                    ]
                }
            ]
            mock_s3_client.get_paginator.return_value = mock_paginator

            files = storage.list_files("s3://bucket/data", "*.parquet")

            assert len(files) == 2
            assert "s3://bucket/data/file1.parquet" in files
            assert "s3://bucket/data/file2.parquet" in files

        def it_returns_empty_list_on_error(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify returns empty list on error."""
            mock_s3_client.get_paginator.side_effect = Exception("Network error")

            files = storage.list_files("s3://bucket/data")

            assert files == []

    class DescribeMakedirs:
        """Tests for the makedirs method."""

        def it_does_nothing_for_s3(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify makedirs is a no-op for S3 (directories are implicit)."""
            storage.makedirs("s3://bucket/path/to/dir")

            # Should not call any S3 methods
            mock_s3_client.put_object.assert_not_called()

    class DescribeGetSizeBytes:
        """Tests for the get_size_bytes method."""

        def it_returns_content_length(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify returns ContentLength from head_object."""
            mock_s3_client.head_object.return_value = {"ContentLength": 12345}

            size = storage.get_size_bytes("s3://bucket/file.txt")

            assert size == 12345

        def it_raises_for_nonexistent_object(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify raises FileDoesNotExistError for missing object."""
            mock_s3_client.head_object.side_effect = Exception("Not found")

            with pytest.raises(FileDoesNotExistError):
                storage.get_size_bytes("s3://bucket/missing.txt")

    class DescribeReadBytes:
        """Tests for the read_bytes method."""

        def it_reads_object_content(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify reads object content as bytes."""
            mock_body = MagicMock()
            mock_body.read.return_value = b"content"
            mock_s3_client.head_object.return_value = {}
            mock_s3_client.get_object.return_value = {"Body": mock_body}

            content = storage.read_bytes("s3://bucket/file.txt")

            assert content == b"content"

        def it_raises_for_nonexistent_object(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify raises FileDoesNotExistError for missing object."""
            from botocore.exceptions import ClientError

            mock_s3_client.head_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "HeadObject"
            )

            with pytest.raises(FileDoesNotExistError):
                storage.read_bytes("s3://bucket/missing.txt")

    class DescribeWriteBytes:
        """Tests for the write_bytes method."""

        def it_writes_bytes_to_object(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify writes bytes using put_object."""
            data = b"content"

            storage.write_bytes(data, "s3://bucket/file.txt")

            mock_s3_client.put_object.assert_called_once_with(
                Bucket="bucket", Key="file.txt", Body=data
            )

        def it_raises_on_error(self, storage: S3StorageService, mock_s3_client: MagicMock) -> None:
            """Verify raises StorageError on write failure."""
            mock_s3_client.put_object.side_effect = Exception("Write failed")

            with pytest.raises(StorageError):
                storage.write_bytes(b"data", "s3://bucket/file.txt")

    class DescribeOpenBinary:
        """Tests for the open_binary context manager."""

        def it_opens_for_reading(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify opens object for reading."""
            mock_body = MagicMock()
            mock_s3_client.head_object.return_value = {}
            mock_s3_client.get_object.return_value = {"Body": mock_body}

            with storage.open_binary("s3://bucket/file.txt", "rb") as f:
                assert f is mock_body

        def it_opens_for_writing(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify opens buffer for writing and uploads on close."""
            with storage.open_binary("s3://bucket/file.txt", "wb") as f:
                f.write(b"content")

            mock_s3_client.put_object.assert_called_once()

        def it_raises_for_unsupported_mode(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify raises for unsupported mode."""
            with pytest.raises(StorageError) as exc_info:
                with storage.open_binary("s3://bucket/file.txt", "a"):
                    pass

            assert "Unsupported mode" in str(exc_info.value)

    class DescribeReadParquet:
        """Tests for the read_parquet method."""

        def it_reads_parquet_from_s3(
            self,
            storage: S3StorageService,
            mock_s3_client: MagicMock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads parquet file from S3."""
            # Create parquet bytes
            buffer = io.BytesIO()
            sample_dataframe.write_parquet(buffer)
            buffer.seek(0)

            # Mock download_file to write the parquet data to a temp file
            def mock_download_file(bucket, key, filename):
                with open(filename, "wb") as f:
                    f.write(buffer.getvalue())

            mock_s3_client.head_object.return_value = {}
            mock_s3_client.download_file.side_effect = mock_download_file

            df = storage.read_parquet("s3://bucket/data.parquet")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

    class DescribeWriteParquet:
        """Tests for the write_parquet method."""

        def it_writes_parquet_to_s3(
            self,
            storage: S3StorageService,
            mock_s3_client: MagicMock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to S3 as parquet."""
            storage.write_parquet(sample_dataframe, "s3://bucket/data.parquet")

            mock_s3_client.put_object.assert_called_once()
            call_args = mock_s3_client.put_object.call_args
            assert call_args[1]["Bucket"] == "bucket"
            assert call_args[1]["Key"] == "data.parquet"

    class DescribeReadJson:
        """Tests for the read_json method."""

        def it_reads_json_from_s3(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify reads JSON from S3."""
            mock_body = MagicMock()
            mock_body.read.return_value = b'{"key": "value"}'
            mock_s3_client.head_object.return_value = {}
            mock_s3_client.get_object.return_value = {"Body": mock_body}

            data = storage.read_json("s3://bucket/data.json")

            assert data == {"key": "value"}

    class DescribeWriteJson:
        """Tests for the write_json method."""

        def it_writes_json_to_s3(
            self, storage: S3StorageService, mock_s3_client: MagicMock
        ) -> None:
            """Verify writes JSON to S3."""
            data = {"key": "value"}

            storage.write_json(data, "s3://bucket/data.json")

            mock_s3_client.put_object.assert_called_once()

    class DescribeLocalCache:
        """Tests for the local_cache context manager."""

        def it_downloads_file_to_cache(
            self, storage: S3StorageService, mock_s3_client: MagicMock, tmp_path: Path
        ) -> None:
            """Verify downloads file to local cache."""
            mock_s3_client.head_object.return_value = {}

            with storage.local_cache("s3://bucket/file.txt") as local_path:
                assert isinstance(local_path, Path)
                mock_s3_client.download_file.assert_called_once()

        def it_cleans_up_cached_file(
            self, storage: S3StorageService, mock_s3_client: MagicMock, tmp_path: Path
        ) -> None:
            """Verify cleans up cached file after context exits."""
            mock_s3_client.head_object.return_value = {}
            # Create a file that would be cleaned up
            mock_s3_client.download_file.side_effect = lambda bucket, key, path: Path(
                path
            ).write_text("content")

            with storage.local_cache("s3://bucket/file.txt") as local_path:
                local_path.write_text("content")  # Ensure file exists
                cached_path = local_path

            assert not cached_path.exists()


class DescribeCreateS3Client:
    """Tests for the create_s3_client factory function."""

    @patch("boto3.Session")
    def it_creates_default_client(self, mock_session_class: MagicMock) -> None:
        """Verify creates client with default credentials."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        create_s3_client()

        mock_session_class.assert_called_once_with()
        mock_session.client.assert_called_once_with("s3")

    @patch("boto3.Session")
    def it_creates_client_with_credentials(self, mock_session_class: MagicMock) -> None:
        """Verify creates client with explicit credentials."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        create_s3_client(
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            region_name="us-west-2",
        )

        mock_session_class.assert_called_once_with(
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            region_name="us-west-2",
        )

    @patch("boto3.Session")
    def it_creates_client_with_endpoint_url(self, mock_session_class: MagicMock) -> None:
        """Verify creates client with custom endpoint (e.g., MinIO)."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        create_s3_client(endpoint_url="http://localhost:9000")

        mock_session.client.assert_called_once_with("s3", endpoint_url="http://localhost:9000")
