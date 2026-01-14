"""Interface for managing storage operations.

This module provides storage abstractions for local filesystem and cloud storage
(AWS S3, Google Cloud Storage). The storage layer uses Polars for DataFrame I/O
and supports transparent local caching for cloud-based joblib memory-mapping.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import polars as pl


@dataclass(frozen=True)
class FileInfo:
    """Metadata about a file in storage."""

    key: str
    size_bytes: int
    last_modified: datetime | None = None


class Storage(Protocol):
    """Interface for storage services.

    Provides a unified interface for file operations across different
    storage backends (local filesystem, S3, GCS).

    Methods cover basic file operations, binary I/O, Polars DataFrame I/O,
    JSON I/O and joblib I/O.
    """

    def exists(self, key: str) -> bool:
        """Check if a file with the given key exists.

        Args:
            key: The key of the file to check.

        Returns:
            True if the file exists, False otherwise.
        """
        ...

    def delete(self, key: str) -> None:
        """Delete a file with the given key.

        Args:
            key: The key of the file to delete.
        Raises:
            StorageError: If there is an error during deletion.
        """
        ...

    def list_files(self, prefix: str, pattern: str = "*") -> Iterator[FileInfo]:
        """List files based on a prefix and optional glob pattern.

        Args:
            prefix: The key prefix to list files under.
            pattern: Optional glob pattern to filter files.

        Returns:
            List of files matching the criteria.
        """
        ...

    def get_size_bytes(self, key: str) -> int:
        """Get the size of a file in bytes.

        Args:
            key: The key of the file.
        Returns:
            Size in bytes.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """
        ...

    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes from a file.

        Args:
            key: The key of the file to read.

        Returns:
            File contents as bytes.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """
        ...

    def write_bytes(self, data: bytes, key: str) -> None:
        """Write raw bytes to a file.

        Args:
            data: The bytes to write.
            key: The key where the file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """
        ...

    def read_parquet(self, key: str) -> pl.DataFrame:
        """Read a parquet file from the given key.

        Args:
            key: The key of the parquet file.

        Returns:
            pl.DataFrame: The DataFrame read from the parquet file.

        Raises:
            FileDoesNotExistError: If the file does not exist at the given key.
        """
        ...

    def write_parquet(self, df: pl.DataFrame, key: str) -> None:
        """Write a DataFrame to a parquet file at the given key.

        Args:
            df: The Polars DataFrame to write.
            key: The key where the parquet file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """
        ...

    def sink_parquet(self, lf: pl.LazyFrame, key: str, **kwargs: Any) -> None:
        """Sink a LazyFrame to a parquet file at the given key.

        This method streams the result to disk without materializing the
        entire DataFrame in memory.

        Args:
            lf: The Polars LazyFrame to sink.
            key: The key where the parquet file will be written.
            **kwargs: Additional arguments passed to sink_parquet.

        Raises:
            StorageError: If there is an error during the write operation.
        """
        ...

    def read_csv(self, key: str, **kwargs: Any) -> pl.DataFrame:
        """Read a CSV file from the given key.

        Args:
            key: The key of the CSV file.
            **kwargs: Additional arguments passed to pl.read_csv.

        Returns:
            pl.DataFrame: The DataFrame read from the CSV file.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """
        ...

    def scan_parquet(self, key: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a parquet file and return a lazy DataFrame.

        This allows for lazy evaluation and memory-efficient processing
        of large datasets without loading the entire file into memory.

        Args:
            key: The key of the parquet file.
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
        ...

    def scan_csv(self, key: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a CSV file and return a lazy DataFrame.

        This allows for lazy evaluation and memory-efficient processing
        of large CSV datasets without loading the entire file into memory.

        Args:
            key: The key of the CSV file.
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
        ...

    def read_json(self, key: str) -> dict[str, Any]:
        """Read a JSON file from the given key.

        Args:
            key: The key of the JSON file.
        Returns:
            Parsed JSON as a dictionary.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """
        ...

    def write_json(self, data: dict[str, Any], key: str) -> None:
        """Write a dictionary to a JSON file at the given key.

        Args:
            data: The dictionary to write.
            key: The key where the JSON file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """
        ...

    def read_joblib(self, key: str, mmap_mode: str | None = None) -> Any:
        """Read a joblib-serialized object from the given key.

        For cloud storage, files are downloaded to a local cache to support
        memory-mapping operations.

        Args:
            key: The key of the joblib file.
            mmap_mode: Memory-map mode ('r', 'r+', 'w+', 'c') or None.

        Returns:
            The deserialized object.

        Raises:
            FileDoesNotExistError: If the file does not exist.
        """

    def write_joblib(self, obj: Any, key: str) -> None:
        """Write an object to a joblib file at the given URI.

        Args:
            obj: The object to serialize.
            key: The key where the joblib file will be written.

        Raises:
            StorageError: If there is an error during the write operation.
        """
        ...
