"""Service for managing file storage operations.

This module provides storage abstractions for local filesystem and cloud storage
(AWS S3, Google Cloud Storage). The storage layer uses Polars for DataFrame I/O
and supports transparent local caching for cloud-based joblib memory-mapping.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Generator
from urllib.parse import urlparse

import polars as pl


class StorageService(ABC):
    """Abstract base class for storage services.

    Provides a unified interface for file operations across different
    storage backends (local filesystem, S3, GCS).

    All URIs should be in the format:
    - file:///path/to/file for local files
    - s3://bucket/path/to/file for S3
    - gs://bucket/path/to/file for GCS
    """

    # --- Core Operations ---

    @abstractmethod
    def construct_uri(self, *parts: str) -> str:
        """Construct a URI from path parts.

        Each storage service implementation should handle its own
        URI construction logic based on its scheme and configuration.

        Args:
            *parts: Path components to join.

        Returns:
            Fully-qualified URI for this storage service.
        """

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Check if a file exists at the given URI.

        Args:
            uri: The URI of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """

    @abstractmethod
    def delete(self, uri: str) -> None:
        """Delete a file at the given URI.

        Args:
            uri: The URI of the file to delete.

        Raises:
            StorageError: If there is an error during deletion.
        """

    @abstractmethod
    def list_files(self, uri: str, pattern: str = "*") -> list[str]:
        """List files in a directory matching a pattern.

        Args:
            uri: The URI of the directory to list.
            pattern: Glob pattern to filter files.

        Returns:
            List of URIs matching the pattern.
        """

    @abstractmethod
    def makedirs(self, uri: str) -> None:
        """Create a directory and all parent directories.

        Args:
            uri: The URI of the directory to create.
        """

    @abstractmethod
    def get_size_bytes(self, uri: str) -> int:
        """Get the size of a file in bytes.

        Args:
            uri: The URI of the file.

        Returns:
            Size in bytes.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """

    # --- Binary I/O ---

    @abstractmethod
    def read_bytes(self, uri: str) -> bytes:
        """Read raw bytes from a file.

        Args:
            uri: The URI of the file to read.

        Returns:
            File contents as bytes.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """

    @abstractmethod
    def write_bytes(self, data: bytes, uri: str) -> None:
        """Write raw bytes to a file.

        Args:
            data: The bytes to write.
            uri: The URI where the file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """

    @abstractmethod
    @contextmanager
    def open_binary(self, uri: str, mode: str = "rb") -> Generator[BinaryIO, None, None]:
        """Open a file for binary I/O.

        Args:
            uri: The URI of the file.
            mode: File mode ('rb' for read, 'wb' for write).

        Yields:
            Binary file handle.
        """

    # --- Polars DataFrame I/O ---

    @abstractmethod
    def read_parquet(self, uri: str) -> pl.DataFrame:
        """Read a parquet file from the given URI.

        Args:
            uri: The URI of the parquet file.

        Returns:
            pl.DataFrame: The DataFrame read from the parquet file.

        Raises:
            FileDoesNotExistError: If the file does not exist at the given URI.
        """

    @abstractmethod
    def write_parquet(self, df: pl.DataFrame, uri: str) -> None:
        """Write a DataFrame to a parquet file at the given URI.

        Args:
            df: The Polars DataFrame to write.
            uri: The URI where the parquet file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """

    @abstractmethod
    def sink_parquet(self, lf: pl.LazyFrame, uri: str, **kwargs: Any) -> None:
        """Sink a LazyFrame to a parquet file at the given URI.

        This method streams the result to disk without materializing the
        entire DataFrame in memory.

        Args:
            lf: The Polars LazyFrame to sink.
            uri: The URI where the parquet file will be written.
            **kwargs: Additional arguments passed to sink_parquet.

        Raises:
            StorageError: If there is an error during the write operation.
        """

    @abstractmethod
    def read_csv(self, uri: str, **kwargs: Any) -> pl.DataFrame:
        """Read a CSV file from the given URI.

        Args:
            uri: The URI of the CSV file.
            **kwargs: Additional arguments passed to pl.read_csv.

        Returns:
            pl.DataFrame: The DataFrame read from the CSV file.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """

    @abstractmethod
    def scan_parquet(self, uri: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a parquet file and return a lazy DataFrame.

        This allows for lazy evaluation and memory-efficient processing
        of large datasets without loading the entire file into memory.

        Args:
            uri: The URI of the parquet file.
            **kwargs: Additional arguments passed to pl.scan_parquet.

        Returns:
            pl.LazyFrame: A lazy DataFrame that can be evaluated later.

        Raises:
            FileDoesNotExistError: If the file does not exist.

        Note:
            For cloud storage (S3/GCS), the file is downloaded to a
            persistent cache directory. The cache must remain available
            during the lazy frame's lifecycle.
        """

    @abstractmethod
    def scan_csv(self, uri: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a CSV file and return a lazy DataFrame.

        This allows for lazy evaluation and memory-efficient processing
        of large CSV datasets without loading the entire file into memory.

        Args:
            uri: The URI of the CSV file.
            **kwargs: Additional arguments passed to pl.scan_csv.

        Returns:
            pl.LazyFrame: A lazy DataFrame that can be evaluated later.

        Raises:
            FileDoesNotExistError: If the file does not exist.

        Note:
            For cloud storage (S3/GCS), the file is downloaded to a
            persistent cache directory. The cache must remain available
            during the lazy frame's lifecycle.
        """

    # --- JSON I/O ---

    @abstractmethod
    def read_json(self, uri: str) -> dict[str, Any]:
        """Read a JSON file from the given URI.

        Args:
            uri: The URI of the JSON file.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """

    @abstractmethod
    def write_json(self, data: dict[str, Any], uri: str) -> None:
        """Write a dictionary to a JSON file at the given URI.

        Args:
            data: The dictionary to write.
            uri: The URI where the JSON file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """

    # --- Joblib I/O (with local caching for memory-mapping) ---

    @abstractmethod
    def read_joblib(self, uri: str, mmap_mode: str | None = None) -> Any:
        """Read a joblib-serialized object from the given URI.

        For cloud storage, files are downloaded to a local cache to support
        memory-mapping operations.

        Args:
            uri: The URI of the joblib file.
            mmap_mode: Memory-map mode ('r', 'r+', 'w+', 'c') or None.

        Returns:
            The deserialized object.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """

    @abstractmethod
    def write_joblib(self, obj: Any, uri: str) -> None:
        """Write an object to a joblib file at the given URI.

        Args:
            obj: The object to serialize.
            uri: The URI where the joblib file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """

    @abstractmethod
    @contextmanager
    def local_cache(self, uri: str) -> Generator[Path, None, None]:
        """Context manager providing a local cached copy of a remote file.

        For local files, yields the path directly.
        For cloud files, downloads to a temp location and yields that path.

        This is useful for operations requiring local filesystem access,
        such as memory-mapping with joblib.

        Args:
            uri: The URI of the file to cache locally.

        Yields:
            Path to the local cached file.
        """

    # --- URI Utilities ---

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str]:
        """Parse a URI into scheme and path components.

        Args:
            uri: The URI to parse.

        Returns:
            Tuple of (scheme, path). Scheme is 'file' for local paths.
            For cloud URIs (s3://, gs://), path includes bucket/key.
        """
        if "://" not in uri:
            # Treat as local path
            return "file", uri

        parsed = urlparse(uri)

        # For file:// URIs, keep the leading slash for absolute paths
        # e.g., file:///tmp/test -> /tmp/test (not tmp/test)
        if parsed.scheme == "file":
            return parsed.scheme, parsed.path
        # For cloud URIs (s3://, gs://), combine netloc (bucket) with path
        # e.g., s3://bucket/path/to/file -> bucket/path/to/file
        if parsed.netloc:
            return parsed.scheme, f"{parsed.netloc}{parsed.path}".lstrip("/")
        # Fall back to just path for other schemes
        return parsed.scheme, parsed.path.lstrip("/")

    @staticmethod
    def to_uri(path: Path | str, scheme: str = "file") -> str:
        """Convert a path to a URI.

        Args:
            path: The path to convert.
            scheme: URI scheme ('file', 's3', 'gs').

        Returns:
            The URI string.
        """
        if isinstance(path, Path):
            path = str(path.absolute())

        if scheme == "file":
            return f"file://{path}"
        return f"{scheme}://{path}"
