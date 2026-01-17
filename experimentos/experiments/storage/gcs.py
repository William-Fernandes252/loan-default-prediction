"""Google Cloud Storage service implementation.

This module provides a GCS-backed storage service that implements the
StorageService interface. It uses google-cloud-storage for GCS operations
and supports transparent local caching for memory-mapping operations.
"""

from collections.abc import Iterator
from datetime import timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

from google.api_core.exceptions import NotFound
from google.cloud.storage import Bucket
import joblib
import polars as pl

from .errors import FileDoesNotExistError, StorageError
from .interface import FileInfo

if TYPE_CHECKING:
    from google.cloud.storage import Client as GCSClient


class GCSStorage:
    """Google Cloud Storage service implementation.

    Uses google-cloud-storage for GCS operations with transparent local
    caching for operations that require local filesystem access (e.g., memory-mapping).

    Keys are treated as object names within the specified bucket and prefix.
    For example, if the bucket is `my-bucket`, the prefix is `data/storage`, and the key is `files/data.csv`, the full GCS path would be `gs://my-bucket/data/storage/files/data.csv`.
    """

    def __init__(
        self,
        gcs_client: "GCSClient",
        bucket_name: str,
        prefix: str = "",
        cache_dir: Path | None = None,
    ):
        self._gcs_client = gcs_client
        self._bucket: Bucket = gcs_client.bucket(bucket_name)
        self._prefix = prefix.strip("/")

        self._cache_dir = cache_dir or Path(tempfile.gettempdir()) / "gcs_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _blob_name(self, key: str) -> str:
        """Construct the full blob name from prefix and key."""
        if self._prefix:
            return f"{self._prefix}/{key}"
        return key

    def _get_cache_path(self, key: str) -> Path:
        """Get the local cache path for a given GCS key."""
        return self._cache_dir / key

    def _download_to_cache(self, key: str) -> Path:
        """Download a GCS object to local cache and return the path."""
        cache_path = self._get_cache_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        blob_name = self._blob_name(key)
        blob = self._bucket.blob(blob_name)

        try:
            blob.download_to_filename(str(cache_path))
            return cache_path
        except NotFound:
            raise FileDoesNotExistError(key)
        except Exception as e:
            raise StorageError(key, str(e))

    def exists(self, key: str) -> bool:
        """Check if a file with the given key exists."""
        try:
            blob_name = self._blob_name(key)
            blob = self._bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            raise StorageError(key, str(e))

    def delete(self, key: str) -> None:
        """Delete a file with the given key."""
        blob_name = self._blob_name(key)
        blob = self._bucket.blob(blob_name)

        try:
            if not blob.exists():
                raise FileDoesNotExistError(key)
            blob.delete()
        except FileDoesNotExistError:
            raise
        except Exception as e:
            raise StorageError(key, str(e))

    def list_files(self, prefix: str, pattern: str = "*") -> Iterator[FileInfo]:
        """List files based on a prefix and optional glob pattern."""
        try:
            full_prefix = self._blob_name(prefix)
            blobs = self._bucket.list_blobs(prefix=full_prefix)

            for blob in blobs:
                # Remove the base prefix to get the relative key
                if self._prefix:
                    key = blob.name[len(self._prefix) + 1 :]
                else:
                    key = blob.name

                # Apply glob pattern matching if not "*"
                if pattern != "*":
                    key_path = Path(key)
                    if not key_path.match(pattern):
                        continue

                yield FileInfo(
                    key=key,
                    size_bytes=blob.size,
                    last_modified=blob.updated.astimezone(timezone.utc) if blob.updated else None,
                )
        except Exception as e:
            raise StorageError(prefix, str(e))

    def get_size_bytes(self, key: str) -> int:
        """Get the size of a file in bytes."""
        blob_name = self._blob_name(key)
        blob = self._bucket.blob(blob_name)

        try:
            blob.reload()
            return int(blob.size or 0)
        except NotFound:
            raise FileDoesNotExistError(key)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes from a file."""
        blob_name = self._blob_name(key)
        blob = self._bucket.blob(blob_name)

        try:
            return blob.download_as_bytes()
        except NotFound:
            raise FileDoesNotExistError(key)
        except Exception as e:
            raise StorageError(key, str(e))

    def write_bytes(self, data: bytes, key: str) -> None:
        """Write raw bytes to a file."""
        blob_name = self._blob_name(key)
        blob = self._bucket.blob(blob_name)

        try:
            blob.upload_from_string(data)
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
            # For GCS, we need to materialize and write
            # since sink_parquet requires a local file path
            cache_path = self._get_cache_path(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            lf.sink_parquet(cache_path, **kwargs)

            # Upload to GCS
            blob_name = self._blob_name(key)
            blob = self._bucket.blob(blob_name)
            blob.upload_from_filename(str(cache_path))

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

            # Upload to GCS
            blob_name = self._blob_name(key)
            blob = self._bucket.blob(blob_name)
            blob.upload_from_filename(str(cache_path))

            # Clean up cache file
            cache_path.unlink(missing_ok=True)
        except Exception as e:
            raise StorageError(key, str(e))


def create_gcs_client(
    *,
    project: str | None = None,
    credentials_file: str | None = None,
) -> "GCSClient":
    """Factory function to create a configured GCS client.

    This function is used by the DI container to create the GCS client
    that will be injected into GCSStorageService.

    Args:
        project: GCP project ID.
        credentials_file: Path to service account JSON file.

    Returns:
        Configured GCS client.
    """
    from google.cloud import storage as gcs_storage

    client_kwargs: dict[str, Any] = {}
    if project:
        client_kwargs["project"] = project

    if credentials_file:
        from google.oauth2 import service_account

        client_kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            credentials_file
        )

    return gcs_storage.Client(**client_kwargs)
