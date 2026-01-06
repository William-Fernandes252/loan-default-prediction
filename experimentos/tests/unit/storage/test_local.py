"""Tests for LocalStorage class."""

import json
from pathlib import Path
import sys

import joblib
import polars as pl
import pytest

# Add parent directory to path to import storage_new modules directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.services.storage_new.errors import FileDoesNotExistError
from experiments.services.storage_new.local import LocalStorage


class DescribeLocalStorage:
    """Tests for the LocalStorage class."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        """Create a LocalStorage instance with temp directory."""
        return LocalStorage(base_path=tmp_path)

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample Polars DataFrame for testing."""
        return pl.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

    class DescribeExists:
        """Tests for the exists method."""

        def it_returns_true_for_existing_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify returns True when file exists."""
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")

            assert storage.exists("test.txt") is True

        def it_returns_false_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify returns False when file doesn't exist."""
            assert storage.exists("nonexistent.txt") is False

        def it_handles_nested_paths(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify works with nested directory paths."""
            nested_dir = tmp_path / "subdir" / "nested"
            nested_dir.mkdir(parents=True)
            test_file = nested_dir / "test.txt"
            test_file.write_text("content")

            assert storage.exists("subdir/nested/test.txt") is True

    class DescribeDelete:
        """Tests for the delete method."""

        def it_deletes_existing_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify file is deleted."""
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")
            assert test_file.exists()

            storage.delete("test.txt")

            assert not test_file.exists()

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.delete("nonexistent.txt")

        def it_deletes_nested_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify deletes file in nested directory."""
            nested_dir = tmp_path / "subdir"
            nested_dir.mkdir()
            test_file = nested_dir / "test.txt"
            test_file.write_text("content")

            storage.delete("subdir/test.txt")

            assert not test_file.exists()

    class DescribeListFiles:
        """Tests for the list_files method."""

        def it_lists_all_files_with_default_pattern(
            self, storage: LocalStorage, tmp_path: Path
        ) -> None:
            """Verify lists all files with default pattern."""
            (tmp_path / "file1.txt").write_text("1")
            (tmp_path / "file2.txt").write_text("2")
            (tmp_path / "file3.csv").write_text("3")

            files = list(storage.list_files(""))

            assert len(files) == 3
            keys = {f.key for f in files}
            assert "file1.txt" in keys
            assert "file2.txt" in keys
            assert "file3.csv" in keys

        def it_filters_by_pattern(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify filters files by glob pattern."""
            (tmp_path / "file1.txt").write_text("1")
            (tmp_path / "file2.txt").write_text("2")
            (tmp_path / "file3.csv").write_text("3")

            files = list(storage.list_files("", "*.txt"))

            assert len(files) == 2
            keys = {f.key for f in files}
            assert "file1.txt" in keys
            assert "file2.txt" in keys

        def it_returns_empty_for_nonexistent_directory(self, storage: LocalStorage) -> None:
            """Verify returns empty iterator for nonexistent directory."""
            files = list(storage.list_files("nonexistent"))

            assert files == []

        def it_includes_file_metadata(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify includes size metadata for files."""
            test_file = tmp_path / "test.txt"
            test_file.write_bytes(b"0" * 100)

            files = list(storage.list_files(""))

            assert len(files) == 1
            assert files[0].size_bytes == 100

        def it_lists_files_in_nested_directory(
            self, storage: LocalStorage, tmp_path: Path
        ) -> None:
            """Verify lists files in nested directories."""
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (subdir / "file1.txt").write_text("1")
            (subdir / "file2.txt").write_text("2")

            files = list(storage.list_files("subdir"))

            assert len(files) == 2

    class DescribeGetSizeBytes:
        """Tests for the get_size_bytes method."""

        def it_returns_file_size(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify returns correct file size in bytes."""
            test_file = tmp_path / "test.txt"
            test_file.write_bytes(b"0" * 100)

            size = storage.get_size_bytes("test.txt")

            assert size == 100

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.get_size_bytes("nonexistent.txt")

    class DescribeReadBytes:
        """Tests for the read_bytes method."""

        def it_reads_file_contents(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify reads file contents as bytes."""
            test_file = tmp_path / "test.bin"
            test_file.write_bytes(b"\x00\x01\x02\x03")

            content = storage.read_bytes("test.bin")

            assert content == b"\x00\x01\x02\x03"

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.read_bytes("nonexistent.bin")

    class DescribeWriteBytes:
        """Tests for the write_bytes method."""

        def it_writes_bytes_to_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify writes bytes to file."""
            data = b"\x00\x01\x02\x03"

            storage.write_bytes(data, "test.bin")

            test_file = tmp_path / "test.bin"
            assert test_file.read_bytes() == data

        def it_creates_parent_directories(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify creates parent directories if needed."""
            data = b"content"

            storage.write_bytes(data, "nested/dir/test.bin")

            test_file = tmp_path / "nested" / "dir" / "test.bin"
            assert test_file.exists()
            assert test_file.read_bytes() == data

    class DescribeReadParquet:
        """Tests for the read_parquet method."""

        def it_reads_parquet_file(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads parquet file into DataFrame."""
            parquet_file = tmp_path / "data.parquet"
            sample_dataframe.write_parquet(parquet_file)

            df = storage.read_parquet("data.parquet")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.read_parquet("nonexistent.parquet")

    class DescribeWriteParquet:
        """Tests for the write_parquet method."""

        def it_writes_dataframe_to_parquet(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to parquet file."""
            storage.write_parquet(sample_dataframe, "data.parquet")

            parquet_file = tmp_path / "data.parquet"
            assert parquet_file.exists()

            df = pl.read_parquet(parquet_file)
            assert df.shape == sample_dataframe.shape

        def it_creates_parent_directories(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify creates parent directories if needed."""
            storage.write_parquet(sample_dataframe, "nested/data.parquet")

            parquet_file = tmp_path / "nested" / "data.parquet"
            assert parquet_file.exists()

    class DescribeSinkParquet:
        """Tests for the sink_parquet method."""

        def it_sinks_lazyframe_to_parquet(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify sinks LazyFrame to parquet file."""
            lf = sample_dataframe.lazy()

            storage.sink_parquet(lf, "data.parquet")

            parquet_file = tmp_path / "data.parquet"
            assert parquet_file.exists()

            df = pl.read_parquet(parquet_file)
            assert df.shape == sample_dataframe.shape

        def it_creates_parent_directories(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify creates parent directories if needed."""
            lf = sample_dataframe.lazy()

            storage.sink_parquet(lf, "nested/dir/data.parquet")

            parquet_file = tmp_path / "nested" / "dir" / "data.parquet"
            assert parquet_file.exists()

    class DescribeReadCsv:
        """Tests for the read_csv method."""

        def it_reads_csv_file(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads CSV file into DataFrame."""
            csv_file = tmp_path / "data.csv"
            sample_dataframe.write_csv(csv_file)

            df = storage.read_csv("data.csv")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.read_csv("nonexistent.csv")

        def it_accepts_kwargs(
            self,
            storage: LocalStorage,
            tmp_path: Path,
        ) -> None:
            """Verify passes kwargs to pl.read_csv."""
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("a;b;c\n1;2;3\n4;5;6")

            df = storage.read_csv("data.csv", separator=";")

            assert df.columns == ["a", "b", "c"]
            assert df.shape == (2, 3)

    class DescribeScanParquet:
        """Tests for the scan_parquet method."""

        def it_scans_parquet_file(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify scans parquet file into LazyFrame."""
            parquet_file = tmp_path / "data.parquet"
            sample_dataframe.write_parquet(parquet_file)

            lf = storage.scan_parquet("data.parquet")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.scan_parquet("nonexistent.parquet")

    class DescribeScanCsv:
        """Tests for the scan_csv method."""

        def it_scans_csv_file(
            self,
            storage: LocalStorage,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify scans CSV file into LazyFrame."""
            csv_file = tmp_path / "data.csv"
            sample_dataframe.write_csv(csv_file)

            lf = storage.scan_csv("data.csv")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.scan_csv("nonexistent.csv")

    class DescribeReadJson:
        """Tests for the read_json method."""

        def it_reads_json_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify reads JSON file into dictionary."""
            json_file = tmp_path / "data.json"
            data = {"key": "value", "number": 42}
            json_file.write_text(json.dumps(data))

            result = storage.read_json("data.json")

            assert result == data

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.read_json("nonexistent.json")

    class DescribeWriteJson:
        """Tests for the write_json method."""

        def it_writes_json_to_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify writes dictionary to JSON file."""
            data = {"key": "value", "number": 42}

            storage.write_json(data, "data.json")

            json_file = tmp_path / "data.json"
            assert json_file.exists()
            result = json.loads(json_file.read_text())
            assert result == data

        def it_creates_parent_directories(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify creates parent directories if needed."""
            data = {"key": "value"}

            storage.write_json(data, "nested/dir/data.json")

            json_file = tmp_path / "nested" / "dir" / "data.json"
            assert json_file.exists()

    class DescribeReadJoblib:
        """Tests for the read_joblib method."""

        def it_reads_joblib_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify reads joblib-serialized object."""
            joblib_file = tmp_path / "data.joblib"
            data = {"key": "value", "list": [1, 2, 3]}
            joblib.dump(data, joblib_file)

            result = storage.read_joblib("data.joblib")

            assert result == data

        def it_raises_for_nonexistent_file(self, storage: LocalStorage) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            with pytest.raises(FileDoesNotExistError):
                storage.read_joblib("nonexistent.joblib")

        def it_supports_mmap_mode(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify supports memory-mapping mode."""
            import numpy as np

            joblib_file = tmp_path / "data.joblib"
            data = np.array([1, 2, 3, 4, 5])
            joblib.dump(data, joblib_file)

            result = storage.read_joblib("data.joblib", mmap_mode="r")

            assert isinstance(result, np.ndarray)
            assert list(result) == [1, 2, 3, 4, 5]

    class DescribeWriteJoblib:
        """Tests for the write_joblib method."""

        def it_writes_joblib_to_file(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify writes object to joblib file."""
            data = {"key": "value", "list": [1, 2, 3]}

            storage.write_joblib(data, "data.joblib")

            joblib_file = tmp_path / "data.joblib"
            assert joblib_file.exists()
            result = joblib.load(joblib_file)
            assert result == data

        def it_creates_parent_directories(self, storage: LocalStorage, tmp_path: Path) -> None:
            """Verify creates parent directories if needed."""
            data = {"key": "value"}

            storage.write_joblib(data, "nested/dir/data.joblib")

            joblib_file = tmp_path / "nested" / "dir" / "data.joblib"
            assert joblib_file.exists()
