"""AWS S3 storage service implementation.

This module provides an S3-backed storage service that implements the
StorageService interface. It uses boto3 for S3 operations and supports
transparent local caching for memory-mapping operations.
"""

from collections.abc import Iterator
from datetime import timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
import joblib
import polars as pl

from .errors import FileDoesNotExistError, StorageError
from .interface import FileInfo

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client


class S3Storage:
    """S3 storage implementation.

    It receives a boto3 S3 client and a bucket name to operate on.

    Keys are treated as object keys within that bucket.
    For example, if the bucket is `my-bucket` and the key is `files/data.csv`,
    the full S3 path would be `s3://my-bucket/files/data.csv`."""

    def __init__(self, s3_client: "S3Client", bucket_name: str, cache_dir: Path | None = None):
        self._s3_client = s3_client
        self._bucket_name = bucket_name
        self._cache_dir = cache_dir or Path(tempfile.gettempdir()) / "s3_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get the local cache path for a given S3 key."""
        return self._cache_dir / key

    def _download_to_cache(self, key: str) -> Path:
        """Download an S3 object to local cache and return the path."""
        cache_path = self._get_cache_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._s3_client.download_file(self._bucket_name, key, str(cache_path))
            return cache_path
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileDoesNotExistError(key)
            raise StorageError(key, str(e))

    def exists(self, key: str) -> bool:
        """Check if a file with the given key exists."""
        try:
            self._s3_client.head_object(Bucket=self._bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise StorageError(key, str(e))

    def delete(self, key: str) -> None:
        """Delete a file with the given key."""
        try:
            # Check if object exists first
            if not self.exists(key):
                raise FileDoesNotExistError(key)

            self._s3_client.delete_object(Bucket=self._bucket_name, Key=key)
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def list_files(self, prefix: str, pattern: str = "*") -> Iterator[FileInfo]:
        """List files based on a prefix and optional glob pattern."""
        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Apply glob pattern matching if not "*"
                    if pattern != "*":
                        key_path = Path(key)
                        if not key_path.match(pattern):
                            continue

                    yield FileInfo(
                        key=key,
                        size_bytes=obj["Size"],
                        last_modified=obj["LastModified"].astimezone(timezone.utc)
                        if obj.get("LastModified")
                        else None,
                    )
        except Exception as e:
            raise StorageError(prefix, str(e))

    def get_size_bytes(self, key: str) -> int:
        """Get the size of a file in bytes."""
        try:
            response = self._s3_client.head_object(Bucket=self._bucket_name, Key=key)
            return response["ContentLength"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileDoesNotExistError(key)
            raise StorageError(key, str(e))

    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes from a file."""
        try:
            response = self._s3_client.get_object(Bucket=self._bucket_name, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileDoesNotExistError(key)
            raise StorageError(key, str(e))

    def write_bytes(self, data: bytes, key: str) -> None:
        """Write raw bytes to a file."""
        try:
            self._s3_client.put_object(Bucket=self._bucket_name, Key=key, Body=data)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_parquet(self, key: str) -> pl.DataFrame:
        """Read a parquet file from the given key."""
        try:
            data = self.read_bytes(key)
            return pl.read_parquet(BytesIO(data))
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def write_parquet(self, df: pl.DataFrame, key: str) -> None:
        """Write a DataFrame to a parquet file at the given key."""
        try:
            buffer = BytesIO()
            df.write_parquet(buffer)
            buffer.seek(0)
            self.write_bytes(buffer.read(), key)
        except Exception as e:
            raise StorageError(key, str(e))

    def sink_parquet(self, lf: pl.LazyFrame, key: str, **kwargs: Any) -> None:
        """Sink a LazyFrame to a parquet file at the given key."""
        try:
            # For S3, we need to materialize and write
            # since sink_parquet requires a local file path
            cache_path = self._get_cache_path(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            lf.sink_parquet(cache_path, **kwargs)

            # Upload to S3
            self._s3_client.upload_file(str(cache_path), self._bucket_name, key)

            # Clean up cache file
            cache_path.unlink(missing_ok=True)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_csv(self, key: str, **kwargs: Any) -> pl.DataFrame:
        """Read a CSV file from the given key."""
        try:
            data = self.read_bytes(key)
            return pl.read_csv(BytesIO(data), **kwargs)
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def scan_parquet(self, key: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a parquet file and return a lazy DataFrame.

        Downloads the file to cache for lazy operations.
        """
        try:
            cache_path = self._download_to_cache(key)
            return pl.scan_parquet(cache_path, **kwargs)
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def scan_csv(self, key: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a CSV file and return a lazy DataFrame.

        Downloads the file to cache for lazy operations.
        """
        try:
            cache_path = self._download_to_cache(key)
            return pl.scan_csv(cache_path, **kwargs)
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def read_json(self, key: str) -> dict[str, Any]:
        """Read a JSON file from the given key."""
        try:
            data = self.read_bytes(key)
            return json.loads(data.decode("utf-8"))
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def write_json(self, data: dict[str, Any], key: str) -> None:
        """Write a dictionary to a JSON file at the given key."""
        try:
            json_bytes = json.dumps(data).encode("utf-8")
            self.write_bytes(json_bytes, key)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_joblib(self, key: str, mmap_mode: str | None = None) -> Any:
        """Read a joblib-serialized object from the given key.

        Downloads the file to cache to support memory-mapping.
        """
        try:
            cache_path = self._download_to_cache(key)
            return joblib.load(cache_path, mmap_mode=mmap_mode)
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def write_joblib(self, obj: Any, key: str) -> None:
        """Write an object to a joblib file at the given key."""
        try:
            cache_path = self._get_cache_path(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(obj, cache_path)

            # Upload to S3
            self._s3_client.upload_file(str(cache_path), self._bucket_name, key)

            # Clean up cache file
            cache_path.unlink(missing_ok=True)
        except Exception as e:
            raise StorageError(key, str(e))
