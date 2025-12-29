"""Google Cloud Storage service implementation.

This module provides a GCS-backed storage service that implements the
StorageService interface. It uses google-cloud-storage for GCS operations
and supports transparent local caching for memory-mapping operations.
"""

from contextlib import contextmanager
import fnmatch
import io
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, BinaryIO, Generator

import joblib
import polars as pl

from experiments.services.storage.base import StorageService
from experiments.services.storage.errors import (
    FileDoesNotExistError,
    StorageError,
)

if TYPE_CHECKING:
    from google.cloud.storage import Client as GCSClient


class GCSStorageService(StorageService):
    """Google Cloud Storage service implementation.

    Uses google-cloud-storage for GCS operations with transparent local
    caching for operations that require local filesystem access (e.g., memory-mapping).

    URIs should be in the format: gs://bucket/path/to/file

    Example:
        ```python
        # Client is injected from the DI container
        storage = GCSStorageService(
            client=gcs_client,
            bucket="my-bucket",
            prefix="experiments/",
        )
        df = storage.read_parquet("gs://my-bucket/experiments/data.parquet")
        ```
    """

    def __init__(
        self,
        client: "GCSClient",
        *,
        bucket: str | None = None,
        prefix: str = "",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the GCS storage service.

        Args:
            client: Pre-configured GCS client (injected dependency).
            bucket: Default bucket name. If provided, paths without bucket
                    will use this bucket.
            prefix: Default prefix to prepend to all paths.
            cache_dir: Directory for local file cache. If None, uses temp directory.
        """
        self._client = client
        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/") if prefix else ""
        self._cache_dir = cache_dir

    def _parse_gcs_path(self, uri: str) -> tuple[str, str]:
        """Parse a URI into bucket and blob name components.

        Returns:
            Tuple of (bucket_name, blob_name).
        """
        scheme, path = self.parse_uri(uri)

        if scheme == "gs":
            # Path is bucket/blob
            parts = path.split("/", 1)
            bucket = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            return bucket, blob_name
        elif scheme == "file":
            raise StorageError(uri, "GCSStorageService does not support file:// URIs")

        # Plain path - use default bucket and prefix
        if self._bucket_name:
            if self._prefix:
                blob_name = f"{self._prefix}/{path.lstrip('/')}"
            else:
                blob_name = path.lstrip("/")
            return self._bucket_name, blob_name
        raise StorageError(uri, "No bucket specified and no default bucket configured")

    def _to_gcs_uri(self, bucket: str, blob_name: str) -> str:
        """Convert bucket and blob name to a full gs:// URI."""
        return f"gs://{bucket}/{blob_name}"

    def _get_bucket(self, bucket_name: str) -> Any:
        """Get a bucket object."""
        return self._client.bucket(bucket_name)

    def _get_blob(self, bucket_name: str, blob_name: str) -> Any:
        """Get a blob object."""
        bucket = self._get_bucket(bucket_name)
        return bucket.blob(blob_name)

    def _blob_exists(self, bucket_name: str, blob_name: str) -> bool:
        """Check if a blob exists in GCS."""
        blob = self._get_blob(bucket_name, blob_name)
        return blob.exists()

    # --- Core Operations ---

    def exists(self, uri: str) -> bool:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        return self._blob_exists(bucket_name, blob_name)

    def delete(self, uri: str) -> None:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        try:
            bucket = self._get_bucket(bucket_name)

            # Check if it's a "directory" (prefix with blobs)
            prefix = blob_name.rstrip("/") + "/"
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))

            if blobs:
                # It's a prefix with blobs - delete all
                blobs_to_delete = bucket.list_blobs(prefix=blob_name)
                for blob in blobs_to_delete:
                    blob.delete()
            else:
                # Single blob
                blob = self._get_blob(bucket_name, blob_name)
                if blob.exists():
                    blob.delete()
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def list_files(self, uri: str, pattern: str = "*") -> list[str]:
        bucket_name, prefix = self._parse_gcs_path(uri)
        prefix = prefix.rstrip("/") + "/" if prefix else ""

        try:
            bucket = self._get_bucket(bucket_name)
            files: list[str] = []

            for blob in bucket.list_blobs(prefix=prefix):
                blob_name = blob.name
                # Get the filename relative to the prefix
                relative_name = blob_name[len(prefix) :] if prefix else blob_name
                # Skip "directory" markers
                if not relative_name or relative_name.endswith("/"):
                    continue
                # Match against the pattern
                if fnmatch.fnmatch(relative_name, pattern):
                    files.append(self._to_gcs_uri(bucket_name, blob_name))

            return files
        except Exception:
            return []

    def makedirs(self, uri: str) -> None:
        # GCS doesn't have real directories, they are created implicitly
        pass

    def get_size_bytes(self, uri: str) -> int:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        blob = self._get_blob(bucket_name, blob_name)
        blob.reload()  # Fetch metadata
        if blob.size is None:
            raise FileDoesNotExistError(uri)
        return blob.size

    # --- Binary I/O ---

    def read_bytes(self, uri: str) -> bytes:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        if not self._blob_exists(bucket_name, blob_name):
            raise FileDoesNotExistError(uri)
        try:
            blob = self._get_blob(bucket_name, blob_name)
            return blob.download_as_bytes()
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def write_bytes(self, data: bytes, uri: str) -> None:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        try:
            blob = self._get_blob(bucket_name, blob_name)
            blob.upload_from_string(data)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    @contextmanager
    def open_binary(self, uri: str, mode: str = "rb") -> Generator[BinaryIO, None, None]:
        bucket_name, blob_name = self._parse_gcs_path(uri)

        if "r" in mode:
            if not self._blob_exists(bucket_name, blob_name):
                raise FileDoesNotExistError(uri)
            try:
                blob = self._get_blob(bucket_name, blob_name)
                data = blob.download_as_bytes()
                yield io.BytesIO(data)
            except Exception as exc:
                raise StorageError(uri, str(exc)) from exc
        elif "w" in mode:
            # For write mode, use a buffer and upload on close
            buffer = io.BytesIO()
            try:
                yield buffer  # type: ignore[misc]
                buffer.seek(0)
                blob = self._get_blob(bucket_name, blob_name)
                blob.upload_from_file(buffer)
            except Exception as exc:
                raise StorageError(uri, str(exc)) from exc
        else:
            raise StorageError(uri, f"Unsupported mode: {mode}")

    # --- Polars DataFrame I/O ---

    def read_parquet(self, uri: str) -> pl.DataFrame:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        if not self._blob_exists(bucket_name, blob_name):
            raise FileDoesNotExistError(uri)
        try:
            blob = self._get_blob(bucket_name, blob_name)
            data = blob.download_as_bytes()
            return pl.read_parquet(io.BytesIO(data))
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def write_parquet(self, df: pl.DataFrame, uri: str) -> None:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        try:
            buffer = io.BytesIO()
            df.write_parquet(buffer)
            buffer.seek(0)
            blob = self._get_blob(bucket_name, blob_name)
            blob.upload_from_file(buffer)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def read_csv(self, uri: str, **kwargs: Any) -> pl.DataFrame:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        if not self._blob_exists(bucket_name, blob_name):
            raise FileDoesNotExistError(uri)
        try:
            blob = self._get_blob(bucket_name, blob_name)
            data = blob.download_as_bytes()
            return pl.read_csv(io.BytesIO(data), **kwargs)
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
        bucket_name, blob_name = self._parse_gcs_path(uri)
        if not self._blob_exists(bucket_name, blob_name):
            raise FileDoesNotExistError(uri)

        if mmap_mode is not None:
            # Memory-mapping requires local file - download to cache
            with self.local_cache(uri) as local_path:
                return joblib.load(local_path, mmap_mode=mmap_mode)
        else:
            # No mmap - can read directly via buffer
            try:
                blob = self._get_blob(bucket_name, blob_name)
                data = blob.download_as_bytes()
                return joblib.load(io.BytesIO(data))
            except Exception as exc:
                raise StorageError(uri, str(exc)) from exc

    def write_joblib(self, obj: Any, uri: str) -> None:
        bucket_name, blob_name = self._parse_gcs_path(uri)
        try:
            # Write to temp file first, then upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                joblib.dump(obj, tmp.name)
                tmp_path = Path(tmp.name)

            try:
                blob = self._get_blob(bucket_name, blob_name)
                blob.upload_from_filename(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    @contextmanager
    def local_cache(self, uri: str) -> Generator[Path, None, None]:
        """Download file to local cache for operations requiring local filesystem.

        For GCS, this downloads the file to a temporary location.
        """
        bucket_name, blob_name = self._parse_gcs_path(uri)
        if not self._blob_exists(bucket_name, blob_name):
            raise FileDoesNotExistError(uri)

        # Determine cache directory
        cache_base = self._cache_dir or Path(tempfile.gettempdir()) / "experiments_cache"
        cache_base.mkdir(parents=True, exist_ok=True)

        # Create a unique local path based on the GCS path
        safe_name = f"{bucket_name}_{blob_name}".replace("/", "_")
        local_path = cache_base / safe_name

        try:
            # Download the file
            blob = self._get_blob(bucket_name, blob_name)
            blob.download_to_filename(str(local_path))
            yield local_path
        finally:
            # Clean up cached file
            local_path.unlink(missing_ok=True)


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
