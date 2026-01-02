from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Any, BinaryIO, Generator

import joblib
import polars as pl

from experiments.services.storage.base import StorageService
from experiments.services.storage.errors import (
    FileDoesNotExistError,
    StorageError,
)


class LocalStorageService(StorageService):
    """Local filesystem storage service implementation.

    Uses file:// URIs for consistency, but also accepts plain paths.
    """

    def _to_path(self, uri: str) -> Path:
        """Convert a URI to a local Path."""
        scheme, path = self.parse_uri(uri)
        if scheme not in ("file", ""):
            raise StorageError(uri, f"LocalStorageService does not support scheme '{scheme}'")
        return Path(path)

    def construct_uri(self, *parts: str) -> str:
        """Construct a file:// URI from path parts."""
        import os

        path = os.path.join(*parts)
        return self.to_uri(Path(path).absolute())

    def exists(self, uri: str) -> bool:
        return self._to_path(uri).exists()

    def delete(self, uri: str) -> None:
        path = self._to_path(uri)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def list_files(self, uri: str, pattern: str = "*") -> list[str]:
        path = self._to_path(uri)
        if not path.exists():
            return []
        return [self.to_uri(p) for p in path.glob(pattern)]

    def makedirs(self, uri: str) -> None:
        path = self._to_path(uri)
        path.mkdir(parents=True, exist_ok=True)

    def get_size_bytes(self, uri: str) -> int:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return path.stat().st_size

    # --- Binary I/O ---

    def read_bytes(self, uri: str) -> bytes:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return path.read_bytes()

    def write_bytes(self, data: bytes, uri: str) -> None:
        path = self._to_path(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    @contextmanager
    def open_binary(self, uri: str, mode: str = "rb") -> Generator[BinaryIO, None, None]:
        path = self._to_path(uri)
        if "w" in mode:
            path.parent.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise FileDoesNotExistError(uri)
        try:
            with open(path, mode) as f:
                yield f  # type: ignore[misc]
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    # --- Polars DataFrame I/O ---

    def read_parquet(self, uri: str) -> pl.DataFrame:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return pl.read_parquet(path)

    def write_parquet(self, df: pl.DataFrame, uri: str) -> None:
        path = self._to_path(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def sink_parquet(self, lf: pl.LazyFrame, uri: str, **kwargs: Any) -> None:
        path = self._to_path(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lf.sink_parquet(path, **kwargs)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    def read_csv(self, uri: str, **kwargs: Any) -> pl.DataFrame:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return pl.read_csv(path, **kwargs)

    def scan_parquet(self, uri: str, **kwargs: Any) -> pl.LazyFrame:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return pl.scan_parquet(path, **kwargs)

    def scan_csv(self, uri: str, **kwargs: Any) -> pl.LazyFrame:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return pl.scan_csv(path, **kwargs)

    # --- JSON I/O ---

    def read_json(self, uri: str) -> dict[str, Any]:
        import json

        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    def write_json(self, data: dict[str, Any], uri: str) -> None:
        import json

        path = self._to_path(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    # --- Joblib I/O ---

    def read_joblib(self, uri: str, mmap_mode: str | None = None) -> Any:
        path = self._to_path(uri)
        if not path.exists():
            raise FileDoesNotExistError(uri)
        return joblib.load(path, mmap_mode=mmap_mode)

    def write_joblib(self, obj: Any, uri: str) -> None:
        path = self._to_path(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, path)
        except Exception as exc:
            raise StorageError(uri, str(exc)) from exc

    @contextmanager
    def local_cache(self, uri: str) -> Generator[Path, None, None]:
        # Local storage: just yield the path directly
        yield self._to_path(uri)
