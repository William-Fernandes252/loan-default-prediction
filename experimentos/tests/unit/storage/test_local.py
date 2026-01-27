from pathlib import Path
from typing import Any

import polars as pl
import pytest

from experiments.storage.errors import FileDoesNotExistError, StorageError
from experiments.storage.interface import FileInfo
from experiments.storage.local import LocalStorage


class DescribeLocalStorageExists:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    def it_returns_true_when_file_exists(self, storage: LocalStorage, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        assert storage.exists("test.txt") is True

    def it_returns_false_when_file_does_not_exist(self, storage: LocalStorage) -> None:
        assert storage.exists("nonexistent.txt") is False

    def it_returns_true_for_nested_file(self, storage: LocalStorage, tmp_path: Path) -> None:
        nested_dir = tmp_path / "dir" / "subdir"
        nested_dir.mkdir(parents=True)
        (nested_dir / "file.txt").write_text("data")

        assert storage.exists("dir/subdir/file.txt") is True


class DescribeLocalStorageDelete:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    def it_deletes_existing_file(self, storage: LocalStorage, tmp_path: Path) -> None:
        file_path = tmp_path / "to_delete.txt"
        file_path.write_text("data")

        storage.delete("to_delete.txt")

        assert not file_path.exists()

    def it_raises_error_when_file_does_not_exist(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError) as exc_info:
            storage.delete("nonexistent.txt")

        assert exc_info.value.uri == "nonexistent.txt"
        assert "File does not exist" in str(exc_info.value)


class DescribeLocalStorageListFiles:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        # Create test file structure
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "file1.csv").write_text("csv1")
        (tmp_path / "data" / "file2.csv").write_text("csv2")
        (tmp_path / "data" / "file.txt").write_text("txt")
        (tmp_path / "data" / "subdir").mkdir()
        (tmp_path / "data" / "subdir" / "file3.csv").write_text("csv3")
        return LocalStorage(tmp_path)

    def it_lists_all_files_with_wildcard(self, storage: LocalStorage) -> None:
        files = list(storage.list_files("data"))

        assert len(files) == 4
        keys = [f.key for f in files]
        assert any("file1.csv" in k for k in keys)
        assert any("file2.csv" in k for k in keys)
        assert any("file.txt" in k for k in keys)
        assert any("file3.csv" in k for k in keys)

    def it_lists_files_matching_pattern(self, storage: LocalStorage) -> None:
        files = list(storage.list_files("data", pattern="*.csv"))

        assert len(files) == 3
        keys = [f.key for f in files]
        assert all(k.endswith(".csv") for k in keys)

    def it_returns_file_info_with_size(self, storage: LocalStorage) -> None:
        files = list(storage.list_files("data", pattern="file1.csv"))

        assert len(files) == 1
        assert isinstance(files[0], FileInfo)
        assert files[0].size_bytes == 4  # "csv1" is 4 bytes

    def it_returns_empty_when_prefix_does_not_exist(self, storage: LocalStorage) -> None:
        files = list(storage.list_files("nonexistent"))

        assert len(files) == 0


class DescribeLocalStorageGetSizeBytes:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    def it_returns_file_size_in_bytes(self, storage: LocalStorage, tmp_path: Path) -> None:
        file_path = tmp_path / "data.bin"
        file_path.write_bytes(b"12345")

        size = storage.get_size_bytes("data.bin")

        assert size == 5

    def it_raises_error_when_file_does_not_exist(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError) as exc_info:
            storage.get_size_bytes("missing.bin")

        assert exc_info.value.uri == "missing.bin"


class DescribeLocalStorageReadWriteBytes:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    def it_writes_and_reads_bytes(self, storage: LocalStorage) -> None:
        data = b"\x00\x01\x02\x03\x04"

        storage.write_bytes(data, "binary.bin")
        result = storage.read_bytes("binary.bin")

        assert result == data

    def it_creates_parent_directories_when_writing(
        self, storage: LocalStorage, tmp_path: Path
    ) -> None:
        storage.write_bytes(b"data", "nested/dir/file.bin")

        assert (tmp_path / "nested" / "dir" / "file.bin").exists()

    def it_raises_error_reading_nonexistent_file(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.read_bytes("missing.bin")


class DescribeLocalStorageReadWriteParquet:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def it_writes_and_reads_parquet(
        self, storage: LocalStorage, sample_dataframe: pl.DataFrame
    ) -> None:
        storage.write_parquet(sample_dataframe, "data.parquet")
        result = storage.read_parquet("data.parquet")

        assert result.equals(sample_dataframe)

    def it_creates_directories_when_writing_parquet(
        self, storage: LocalStorage, sample_dataframe: pl.DataFrame, tmp_path: Path
    ) -> None:
        storage.write_parquet(sample_dataframe, "output/data.parquet")

        assert (tmp_path / "output" / "data.parquet").exists()

    def it_raises_error_reading_nonexistent_parquet(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.read_parquet("missing.parquet")


class DescribeLocalStorageSinkParquet:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    @pytest.fixture
    def sample_lazyframe(self) -> pl.LazyFrame:
        return pl.LazyFrame({"x": [10, 20, 30], "y": [100, 200, 300]})

    def it_sinks_lazyframe_to_parquet(
        self, storage: LocalStorage, sample_lazyframe: pl.LazyFrame
    ) -> None:
        storage.sink_parquet(sample_lazyframe, "lazy.parquet")

        result = storage.read_parquet("lazy.parquet")
        assert result.shape == (3, 2)
        assert result.columns == ["x", "y"]

    def it_creates_directories_when_sinking(
        self, storage: LocalStorage, sample_lazyframe: pl.LazyFrame, tmp_path: Path
    ) -> None:
        storage.sink_parquet(sample_lazyframe, "nested/lazy.parquet")

        assert (tmp_path / "nested" / "lazy.parquet").exists()


class DescribeLocalStorageReadCsv:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("name,age\nAlice,30\nBob,25")
        return LocalStorage(tmp_path)

    def it_reads_csv_file(self, storage: LocalStorage) -> None:
        df = storage.read_csv("data.csv")

        assert df.shape == (2, 2)
        assert df.columns == ["name", "age"]

    def it_raises_error_when_csv_does_not_exist(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.read_csv("missing.csv")


class DescribeLocalStorageScanParquet:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        parquet_path = tmp_path / "scan.parquet"
        df.write_parquet(parquet_path)
        return LocalStorage(tmp_path)

    def it_scans_parquet_as_lazyframe(self, storage: LocalStorage) -> None:
        lf = storage.scan_parquet("scan.parquet")

        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert df.shape == (3, 2)

    def it_raises_error_when_parquet_does_not_exist(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.scan_parquet("missing.parquet")


class DescribeLocalStorageScanCsv:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        csv_path = tmp_path / "scan.csv"
        csv_path.write_text("id,value\n1,100\n2,200")
        return LocalStorage(tmp_path)

    def it_scans_csv_as_lazyframe(self, storage: LocalStorage) -> None:
        lf = storage.scan_csv("scan.csv")

        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert df.shape == (2, 2)

    def it_raises_error_when_csv_does_not_exist(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.scan_csv("missing.csv")


class DescribeLocalStorageReadWriteJson:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    @pytest.fixture
    def sample_json(self) -> dict[str, Any]:
        return {"name": "test", "value": 42, "items": [1, 2, 3]}

    def it_writes_and_reads_json(self, storage: LocalStorage, sample_json: dict[str, Any]) -> None:
        storage.write_json(sample_json, "config.json")
        result = storage.read_json("config.json")

        assert result == sample_json

    def it_creates_directories_when_writing_json(
        self, storage: LocalStorage, sample_json: dict[str, Any], tmp_path: Path
    ) -> None:
        storage.write_json(sample_json, "configs/app.json")

        assert (tmp_path / "configs" / "app.json").exists()

    def it_raises_error_reading_nonexistent_json(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.read_json("missing.json")


class DescribeLocalStorageReadWriteJoblib:
    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        return LocalStorage(tmp_path)

    @pytest.fixture
    def sample_object(self) -> dict[str, Any]:
        return {"model": "test", "params": {"alpha": 0.5, "beta": 0.3}}

    def it_writes_and_reads_joblib(
        self, storage: LocalStorage, sample_object: dict[str, Any]
    ) -> None:
        storage.write_joblib(sample_object, "model.joblib")
        result = storage.read_joblib("model.joblib")

        assert result == sample_object

    def it_creates_directories_when_writing_joblib(
        self, storage: LocalStorage, sample_object: dict[str, Any], tmp_path: Path
    ) -> None:
        storage.write_joblib(sample_object, "models/trained.joblib")

        assert (tmp_path / "models" / "trained.joblib").exists()

    def it_raises_error_reading_nonexistent_joblib(self, storage: LocalStorage) -> None:
        with pytest.raises(FileDoesNotExistError):
            storage.read_joblib("missing.joblib")


class DescribeStorageErrors:
    def it_storage_error_contains_uri_and_reason(self) -> None:
        error = StorageError("test/path", "Connection failed")

        assert error.uri == "test/path"
        assert error.reason == "Connection failed"
        assert "test/path" in str(error)
        assert "Connection failed" in str(error)

    def it_file_does_not_exist_error_inherits_from_storage_error(self) -> None:
        error = FileDoesNotExistError("missing/file.txt")

        assert isinstance(error, StorageError)
        assert error.uri == "missing/file.txt"
        assert "File does not exist" in str(error)
