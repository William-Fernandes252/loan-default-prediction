from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from .errors import FileDoesNotExistError, StorageError
from .interface import FileInfo


class LocalStorage:
    """Local storage implementation.

    It receives a base path where files are stored, and uses that as root for all operations.

    Keys are treated as relative paths from that base path.
    For example, if the base path is `/data/storage` and the key is `files/data.csv`,
    the full path would be `/data/storage/files/data.csv`.
    """

    def __init__(self, base_path: Path):
        self._base_path = base_path

    def _full_path(self, key: str) -> Path:
        return self._base_path / key

    def exists(self, key: str) -> bool:
        """Check if a file with the given key exists."""
        return self._full_path(key).exists()

    def delete(self, key: str) -> None:
        """Delete a file with the given key."""
        path = self._full_path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            raise FileDoesNotExistError(key)
        except Exception as e:
            raise StorageError(key, str(e))

    def list_files(self, prefix: str, pattern: str = "*") -> Iterator[FileInfo]:
        """List files recursively based on a prefix and optional glob pattern."""
        prefix_path = self._full_path(prefix)
        if not prefix_path.exists():
            return

        for path in prefix_path.rglob(pattern):
            if path.is_file():
                stat = path.stat()
                key = str(path.relative_to(self._base_path))
                yield FileInfo(
                    key=key,
                    size_bytes=stat.st_size,
                    last_modified=None,  # Can be added if needed
                )

    def get_size_bytes(self, key: str) -> int:
        """Get the size of a file in bytes."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        return path.stat().st_size

    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes from a file."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            return path.read_bytes()
        except Exception as e:
            raise StorageError(key, str(e))

    def write_bytes(self, data: bytes, key: str) -> None:
        """Write raw bytes to a file."""
        path = self._full_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_parquet(self, key: str) -> pl.DataFrame:
        """Read a parquet file from the given key."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            return pl.read_parquet(path)
        except Exception as e:
            raise StorageError(key, str(e))

    def write_parquet(self, df: pl.DataFrame, key: str) -> None:
        """Write a DataFrame to a parquet file at the given key."""
        path = self._full_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
        except Exception as e:
            raise StorageError(key, str(e))

    def sink_parquet(self, lf: pl.LazyFrame, key: str, **kwargs: Any) -> None:
        """Sink a LazyFrame to a parquet file at the given key."""
        path = self._full_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lf.sink_parquet(path, **kwargs)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_csv(self, key: str, **kwargs: Any) -> pl.DataFrame:
        """Read a CSV file from the given key."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            return pl.read_csv(path, **kwargs)
        except Exception as e:
            raise StorageError(key, str(e))

    def scan_parquet(self, key: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a parquet file and return a lazy DataFrame."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            return pl.scan_parquet(path, **kwargs)
        except Exception as e:
            raise StorageError(key, str(e))

    def scan_csv(self, key: str, **kwargs: Any) -> pl.LazyFrame:
        """Scan a CSV file and return a lazy DataFrame."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            return pl.scan_csv(path, **kwargs)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_json(self, key: str) -> dict[str, Any]:
        """Read a JSON file from the given key."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            with path.open() as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(key, str(e))

    def write_json(self, data: dict[str, Any], key: str) -> None:
        """Write a dictionary to a JSON file at the given key."""
        path = self._full_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(data, f)
        except Exception as e:
            raise StorageError(key, str(e))

    def read_joblib(self, key: str, mmap_mode: str | None = None) -> Any:
        """Read a joblib-serialized object from the given key."""
        path = self._full_path(key)
        if not path.exists():
            raise FileDoesNotExistError(key)
        try:
            return joblib.load(path, mmap_mode=mmap_mode)
        except Exception as e:
            raise StorageError(key, str(e))

    def write_joblib(self, obj: Any, key: str) -> None:
        """Write an object to a joblib file at the given key."""
        path = self._full_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, path)
        except Exception as e:
            raise StorageError(key, str(e))
