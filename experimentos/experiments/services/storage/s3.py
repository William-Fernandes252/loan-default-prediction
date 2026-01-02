"""AWS S3 storage service implementation.

This module provides an S3-backed storage service that implements the
StorageService interface. It uses boto3 for S3 operations and supports
transparent local caching for memory-mapping operations.
"""

from contextlib import contextmanager
import fnmatch
import io
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, BinaryIO, Generator, cast

import joblib
import polars as pl

from experiments.services.storage.base import StorageService
from experiments.services.storage.errors import (
    FileDoesNotExistError,
    StorageError,
)

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client


class S3StorageService(StorageService):
    """AWS S3 storage service implementation.

    Uses boto3 for S3 operations with transparent local caching for
    operations that require local filesystem access (e.g., memory-mapping).

    URIs should be in the format: s3://bucket/path/to/file

    Example:
        ```python
        # Client is injected from the DI container
        storage = S3StorageService(
            client=boto3_client,
            bucket="my-bucket",
            prefix="experiments/",
        )
        df = storage.read_parquet("s3://my-bucket/experiments/data.parquet")
        ```
    """

    def __init__(
        self,
        client: "S3Client",
        *,
        bucket: str | None = None,
        prefix: str = "",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the S3 storage service.

        Args:
            client: Pre-configured boto3 S3 client (injected dependency).
            bucket: Default bucket name. If provided, paths without bucket
                    will use this bucket.
            prefix: Default prefix to prepend to all paths.
            cache_dir: Directory for local file cache. If None, uses temp directory.
        """
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") if prefix else ""
        self._cache_dir = cache_dir

    def _parse_s3_path(self, uri: str) -> tuple[str, str]:
        """Parse a URI into bucket and key components.

        Returns:
            Tuple of (bucket, key).
        """
        scheme, path = self.parse_uri(uri)

        if scheme == "s3":
            # Path is bucket/key
            parts = path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return bucket, key
        elif scheme == "file":
            raise StorageError(uri, "S3StorageService does not support file:// URIs")

        # Plain path - use default bucket and prefix
        if self._bucket:
            if self._prefix:
                key = f"{self._prefix}/{path.lstrip('/')}"
            else:
                key = path.lstrip("/")
            return self._bucket, key
        raise StorageError(uri, "No bucket specified and no default bucket configured")

    def _to_s3_uri(self, bucket: str, key: str) -> str:
        """Convert bucket and key to a full s3:// URI."""
        return f"s3://{bucket}/{key}"

    def _object_exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists in S3."""
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    # --- Core Operations ---

    def construct_uri(self, *parts: str) -> str:
        """Construct an s3:// URI from path parts."""
        import os

        path = os.path.join(*parts)

        # Use configured bucket
        bucket = self._bucket
        if not bucket:
            raise ValueError("S3 bucket must be configured")

        # Apply prefix if configured
        if self._prefix:
            path = f"{self._prefix}/{path.lstrip('/')}"

        clean_path = path.lstrip("/")
        return f"s3://{bucket}/{clean_path}"

    def exists(self, uri: str) -> bool:
        bucket, key = self._parse_s3_path(uri)
        return self._object_exists(bucket, key)

    def delete(self, uri: str) -> None:
        bucket, key = self._parse_s3_path(uri)
        try:
            # Check if it's a "directory" (prefix with objects)
            response = self._client.list_objects_v2(
                Bucket=bucket, Prefix=key.rstrip("/") + "/", MaxKeys=1
            )
            if response.get("Contents"):
                # It's a prefix with objects - delete all
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=key):
                    if "Contents" in page:
                        objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                        self._client.delete_objects(Bucket=bucket, Delete={"Objects": objects})  # type: ignore[typeddict-item]
            else:
                # Single object
                self._client.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def list_files(self, uri: str, pattern: str = "*") -> list[str]:
        bucket, prefix = self._parse_s3_path(uri)
        prefix = prefix.rstrip("/") + "/" if prefix else ""

        try:
            paginator = self._client.get_paginator("list_objects_v2")
            files: list[str] = []

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Get the filename relative to the prefix
                    relative_name = key[len(prefix) :] if prefix else key
                    # Match against the pattern
                    if fnmatch.fnmatch(relative_name, pattern):
                        files.append(self._to_s3_uri(bucket, key))

            return files
        except Exception:
            return []

    def makedirs(self, uri: str) -> None:
        # S3 doesn't have real directories, they are created implicitly
        pass

    def get_size_bytes(self, uri: str) -> int:
        bucket, key = self._parse_s3_path(uri)
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
            return response["ContentLength"]
        except Exception:
            raise FileDoesNotExistError(uri)

    # --- Binary I/O ---

    def read_bytes(self, uri: str) -> bytes:
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def write_bytes(self, data: bytes, uri: str) -> None:
        bucket, key = self._parse_s3_path(uri)
        try:
            self._client.put_object(Bucket=bucket, Key=key, Body=data)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    @contextmanager
    def open_binary(self, uri: str, mode: str = "rb") -> Generator[BinaryIO, None, None]:
        bucket, key = self._parse_s3_path(uri)

        if "r" in mode:
            if not self._object_exists(bucket, key):
                raise FileDoesNotExistError(uri)
            try:
                response = self._client.get_object(Bucket=bucket, Key=key)
                yield cast(BinaryIO, response["Body"])
            except Exception as exc:
                raise StorageError(uri, str(exc)) from exc
        elif "w" in mode:
            # For write mode, use a buffer and upload on close
            buffer = io.BytesIO()
            try:
                yield buffer  # type: ignore[misc]
                buffer.seek(0)
                self._client.put_object(Bucket=bucket, Key=key, Body=buffer.read())
            except Exception as exc:
                raise StorageError(uri, str(exc)) from exc
        else:
            raise StorageError(uri, f"Unsupported mode: {mode}")

    # --- Polars DataFrame I/O ---

    def read_parquet(self, uri: str) -> pl.DataFrame:
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)
        try:
            # Download to a temporary file to avoid loading entire file into RAM
            # This allows Polars to use memory-mapped I/O for efficient reading
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
                self._client.download_file(bucket, key, tmp.name)
                return pl.read_parquet(tmp.name)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def write_parquet(self, df: pl.DataFrame, uri: str) -> None:
        bucket, key = self._parse_s3_path(uri)
        try:
            buffer = io.BytesIO()
            df.write_parquet(buffer)
            buffer.seek(0)
            self._client.put_object(Bucket=bucket, Key=key, Body=buffer.read())
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def sink_parquet(self, lf: pl.LazyFrame, uri: str, **kwargs: Any) -> None:
        bucket, key = self._parse_s3_path(uri)
        try:
            # Sink to a temp file first to keep memory usage low
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
                lf.sink_parquet(tmp.name, **kwargs)
                self._client.upload_file(tmp.name, bucket, key)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def read_csv(self, uri: str, **kwargs: Any) -> pl.DataFrame:
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)
        try:
            # Download to a temporary file to avoid loading entire file into RAM
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp:
                self._client.download_file(bucket, key, tmp.name)
                return pl.read_csv(tmp.name, **kwargs)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def scan_parquet(self, uri: str, **kwargs: Any) -> pl.LazyFrame:
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)
        try:
            # Download to persistent cache for lazy evaluation
            # The cache must persist during the lazy frame's lifecycle
            cache_base = self._cache_dir or Path(tempfile.gettempdir()) / "experiments_cache"
            cache_base.mkdir(parents=True, exist_ok=True)

            # Create a unique local path based on the S3 path
            safe_name = f"{bucket}_{key}".replace("/", "_")
            local_path = cache_base / safe_name

            # Download if not already cached
            if not local_path.exists():
                self._client.download_file(bucket, key, str(local_path))

            return pl.scan_parquet(local_path, **kwargs)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def scan_csv(self, uri: str, **kwargs: Any) -> pl.LazyFrame:
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)
        try:
            # Download to persistent cache for lazy evaluation
            # The cache must persist during the lazy frame's lifecycle
            cache_base = self._cache_dir or Path(tempfile.gettempdir()) / "experiments_cache"
            cache_base.mkdir(parents=True, exist_ok=True)

            # Create a unique local path based on the S3 path
            safe_name = f"{bucket}_{key}".replace("/", "_")
            local_path = cache_base / safe_name

            # Download if not already cached
            if not local_path.exists():
                self._client.download_file(bucket, key, str(local_path))

            return pl.scan_csv(local_path, **kwargs)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    # --- JSON I/O ---

    def read_json(self, uri: str) -> dict[str, Any]:
        import json

        data = self.read_bytes(uri)
        return json.loads(data.decode("utf-8"))  # type: ignore[no-any-return]

    def write_json(self, data: dict[str, Any], uri: str) -> None:
        import json

        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.write_bytes(json_bytes, uri)

    # --- Joblib I/O (with local caching for memory-mapping) ---

    def read_joblib(self, uri: str, mmap_mode: str | None = None) -> Any:
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)

        if mmap_mode is not None:
            # Memory-mapping requires local file - download to cache
            with self.local_cache(uri) as local_path:
                return joblib.load(local_path, mmap_mode=mmap_mode)
        else:
            # No mmap - can read directly via streaming
            try:
                response = self._client.get_object(Bucket=bucket, Key=key)
                buffer = io.BytesIO(response["Body"].read())
                return joblib.load(buffer)
            except Exception as exc:
                raise StorageError(uri, str(exc)) from exc

    def write_joblib(self, obj: Any, uri: str) -> None:
        bucket, key = self._parse_s3_path(uri)
        try:
            # Write to temp file first, then upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                joblib.dump(obj, tmp.name)
                tmp_path = Path(tmp.name)

            try:
                with open(tmp_path, "rb") as f:
                    self._client.put_object(Bucket=bucket, Key=key, Body=f.read())
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    @contextmanager
    def local_cache(self, uri: str) -> Generator[Path, None, None]:
        """Download file to local cache for operations requiring local filesystem.

        For S3, this downloads the file to a temporary location.
        """
        bucket, key = self._parse_s3_path(uri)
        if not self._object_exists(bucket, key):
            raise FileDoesNotExistError(uri)

        # Determine cache directory
        cache_base = self._cache_dir or Path(tempfile.gettempdir()) / "experiments_cache"
        cache_base.mkdir(parents=True, exist_ok=True)

        # Create a unique local path based on the S3 path
        safe_name = f"{bucket}_{key}".replace("/", "_")
        local_path = cache_base / safe_name

        try:
            # Download the file
            self._client.download_file(bucket, key, str(local_path))
            yield local_path
        finally:
            # Clean up cached file
            local_path.unlink(missing_ok=True)


def create_s3_client(
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    region_name: str | None = None,
    endpoint_url: str | None = None,
) -> "S3Client":
    """Factory function to create a configured boto3 S3 client.

    This function is used by the DI container to create the S3 client
    that will be injected into S3StorageService.

    Args:
        aws_access_key_id: AWS access key ID. If None, uses environment/credentials.
        aws_secret_access_key: AWS secret access key.
        aws_session_token: AWS session token for temporary credentials.
        region_name: AWS region name.
        endpoint_url: Custom endpoint URL (for S3-compatible services).

    Returns:
        Configured boto3 S3 client.
    """
    import boto3

    # Build boto3 session
    session_kwargs: dict[str, Any] = {}
    if aws_access_key_id:
        session_kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        session_kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        session_kwargs["aws_session_token"] = aws_session_token
    if region_name:
        session_kwargs["region_name"] = region_name

    session = boto3.Session(**session_kwargs)

    # Build client with optional endpoint
    client_kwargs: dict[str, Any] = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    return session.client("s3", **client_kwargs)
