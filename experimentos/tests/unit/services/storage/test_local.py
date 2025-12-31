import json
from pathlib import Path

import joblib
import polars as pl
import pytest

from experiments.services.storage import FileDoesNotExistError, LocalStorageService, StorageError


class DescribeLocalStorageService:
    """Tests for the LocalStorageService class."""

    @pytest.fixture
    def storage(self) -> LocalStorageService:
        """Create a LocalStorageService instance."""
        return LocalStorageService()

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

        def it_returns_true_for_existing_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify returns True when file exists."""
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")

            assert storage.exists(f"file://{test_file}") is True

        def it_returns_false_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify returns False when file doesn't exist."""
            nonexistent = tmp_path / "nonexistent.txt"

            assert storage.exists(f"file://{nonexistent}") is False

        def it_accepts_plain_path(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify accepts plain path without file:// prefix."""
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")

            assert storage.exists(str(test_file)) is True

    class DescribeDelete:
        """Tests for the delete method."""

        def it_deletes_existing_file(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify file is deleted."""
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")
            assert test_file.exists()

            storage.delete(f"file://{test_file}")

            assert not test_file.exists()

        def it_deletes_directory_recursively(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify directory and contents are deleted."""
            test_dir = tmp_path / "subdir"
            test_dir.mkdir()
            (test_dir / "file.txt").write_text("content")

            storage.delete(f"file://{test_dir}")

            assert not test_dir.exists()

        def it_handles_nonexistent_file_gracefully(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify no error when file doesn't exist."""
            nonexistent = tmp_path / "nonexistent.txt"

            storage.delete(f"file://{nonexistent}")  # Should not raise

    class DescribeListFiles:
        """Tests for the list_files method."""

        def it_lists_all_files(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify lists all files in directory."""
            (tmp_path / "file1.txt").write_text("1")
            (tmp_path / "file2.txt").write_text("2")
            (tmp_path / "file3.csv").write_text("3")

            files = storage.list_files(f"file://{tmp_path}")

            assert len(files) == 3

        def it_filters_by_pattern(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify filters files by glob pattern."""
            (tmp_path / "file1.txt").write_text("1")
            (tmp_path / "file2.txt").write_text("2")
            (tmp_path / "file3.csv").write_text("3")

            files = storage.list_files(f"file://{tmp_path}", "*.txt")

            assert len(files) == 2
            assert all(".txt" in f for f in files)

        def it_returns_empty_for_nonexistent_directory(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify returns empty list for nonexistent directory."""
            nonexistent = tmp_path / "nonexistent"

            files = storage.list_files(f"file://{nonexistent}")

            assert files == []

    class DescribeMakedirs:
        """Tests for the makedirs method."""

        def it_creates_directory(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify directory is created."""
            new_dir = tmp_path / "new" / "nested" / "dir"

            storage.makedirs(f"file://{new_dir}")

            assert new_dir.exists()
            assert new_dir.is_dir()

        def it_handles_existing_directory(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify no error when directory already exists."""
            existing_dir = tmp_path / "existing"
            existing_dir.mkdir()

            storage.makedirs(f"file://{existing_dir}")  # Should not raise

    class DescribeGetSizeBytes:
        """Tests for the get_size_bytes method."""

        def it_returns_file_size(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify returns correct file size in bytes."""
            test_file = tmp_path / "test.txt"
            test_file.write_bytes(b"0" * 100)

            size = storage.get_size_bytes(f"file://{test_file}")

            assert size == 100

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.txt"

            with pytest.raises(FileDoesNotExistError):
                storage.get_size_bytes(f"file://{nonexistent}")

    class DescribeReadBytes:
        """Tests for the read_bytes method."""

        def it_reads_file_contents(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify reads file contents as bytes."""
            test_file = tmp_path / "test.bin"
            test_file.write_bytes(b"\x00\x01\x02\x03")

            content = storage.read_bytes(f"file://{test_file}")

            assert content == b"\x00\x01\x02\x03"

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.bin"

            with pytest.raises(FileDoesNotExistError):
                storage.read_bytes(f"file://{nonexistent}")

    class DescribeWriteBytes:
        """Tests for the write_bytes method."""

        def it_writes_bytes_to_file(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify writes bytes to file."""
            test_file = tmp_path / "test.bin"
            data = b"\x00\x01\x02\x03"

            storage.write_bytes(data, f"file://{test_file}")

            assert test_file.read_bytes() == data

        def it_creates_parent_directories(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify creates parent directories if needed."""
            test_file = tmp_path / "nested" / "dir" / "test.bin"
            data = b"content"

            storage.write_bytes(data, f"file://{test_file}")

            assert test_file.exists()

    class DescribeOpenBinary:
        """Tests for the open_binary context manager."""

        def it_opens_file_for_reading(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify opens file in read mode."""
            test_file = tmp_path / "test.bin"
            test_file.write_bytes(b"content")

            with storage.open_binary(f"file://{test_file}", "rb") as f:
                content = f.read()

            assert content == b"content"

        def it_opens_file_for_writing(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify opens file in write mode."""
            test_file = tmp_path / "test.bin"

            with storage.open_binary(f"file://{test_file}", "wb") as f:
                f.write(b"content")

            assert test_file.read_bytes() == b"content"

        def it_raises_for_nonexistent_read(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError when reading missing file."""
            nonexistent = tmp_path / "nonexistent.bin"

            with pytest.raises(FileDoesNotExistError):
                with storage.open_binary(f"file://{nonexistent}", "rb"):
                    pass

    class DescribeReadParquet:
        """Tests for the read_parquet method."""

        def it_reads_parquet_file(
            self,
            storage: LocalStorageService,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads parquet file into DataFrame."""
            parquet_file = tmp_path / "data.parquet"
            sample_dataframe.write_parquet(parquet_file)

            df = storage.read_parquet(f"file://{parquet_file}")

            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.parquet"

            with pytest.raises(FileDoesNotExistError):
                storage.read_parquet(f"file://{nonexistent}")

    class DescribeWriteParquet:
        """Tests for the write_parquet method."""

        def it_writes_dataframe_to_parquet(
            self,
            storage: LocalStorageService,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to parquet file."""
            parquet_file = tmp_path / "data.parquet"

            storage.write_parquet(sample_dataframe, f"file://{parquet_file}")

            assert parquet_file.exists()
            df = pl.read_parquet(parquet_file)
            assert df.shape == sample_dataframe.shape

        def it_creates_parent_directories(
            self,
            storage: LocalStorageService,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify creates parent directories if needed."""
            parquet_file = tmp_path / "nested" / "data.parquet"

            storage.write_parquet(sample_dataframe, f"file://{parquet_file}")

            assert parquet_file.exists()

    class DescribeReadCsv:
        """Tests for the read_csv method."""

        def it_reads_csv_file(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify reads CSV file into DataFrame."""
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

            df = storage.read_csv(f"file://{csv_file}")

            assert df.shape == (2, 3)
            assert df.columns == ["a", "b", "c"]

        def it_passes_kwargs_to_polars(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify passes additional kwargs to pl.read_csv."""
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("a;b;c\n1;2;3")

            df = storage.read_csv(f"file://{csv_file}", separator=";")

            assert df.shape == (1, 3)

    class DescribeScanParquet:
        """Tests for the scan_parquet method."""

        def it_returns_lazy_frame(
            self,
            storage: LocalStorageService,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify returns a lazy DataFrame."""
            parquet_file = tmp_path / "data.parquet"
            sample_dataframe.write_parquet(parquet_file)

            lf = storage.scan_parquet(f"file://{parquet_file}")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == sample_dataframe.shape
            assert df.columns == sample_dataframe.columns

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.parquet"

            with pytest.raises(FileDoesNotExistError):
                storage.scan_parquet(f"file://{nonexistent}")

        def it_passes_kwargs_to_polars(
            self,
            storage: LocalStorageService,
            tmp_path: Path,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify passes additional kwargs to pl.scan_parquet."""
            parquet_file = tmp_path / "data.parquet"
            sample_dataframe.write_parquet(parquet_file)

            lf = storage.scan_parquet(f"file://{parquet_file}", n_rows=2)
            df = lf.collect()

            assert df.shape == (2, 3)

    class DescribeScanCsv:
        """Tests for the scan_csv method."""

        def it_returns_lazy_frame(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify returns a lazy DataFrame."""
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

            lf = storage.scan_csv(f"file://{csv_file}")

            assert isinstance(lf, pl.LazyFrame)
            df = lf.collect()
            assert df.shape == (2, 3)
            assert df.columns == ["a", "b", "c"]

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.csv"

            with pytest.raises(FileDoesNotExistError):
                storage.scan_csv(f"file://{nonexistent}")

        def it_passes_kwargs_to_polars(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify passes additional kwargs to pl.scan_csv."""
            csv_file = tmp_path / "data.csv"
            csv_file.write_text("a;b;c\n1;2;3\n4;5;6")

            lf = storage.scan_csv(f"file://{csv_file}", separator=";")
            df = lf.collect()

            assert df.shape == (2, 3)

    class DescribeReadJson:
        """Tests for the read_json method."""

        def it_reads_json_file(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify reads JSON file into dictionary."""
            json_file = tmp_path / "data.json"
            json_file.write_text('{"key": "value", "number": 42}')

            data = storage.read_json(f"file://{json_file}")

            assert data == {"key": "value", "number": 42}

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.json"

            with pytest.raises(FileDoesNotExistError):
                storage.read_json(f"file://{nonexistent}")

    class DescribeWriteJson:
        """Tests for the write_json method."""

        def it_writes_dictionary_to_json(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify writes dictionary to JSON file."""
            json_file = tmp_path / "data.json"
            data = {"key": "value", "number": 42}

            storage.write_json(data, f"file://{json_file}")

            assert json_file.exists()
            assert json.loads(json_file.read_text()) == data

        def it_creates_parent_directories(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify creates parent directories if needed."""
            json_file = tmp_path / "nested" / "data.json"
            data = {"key": "value"}

            storage.write_json(data, f"file://{json_file}")

            assert json_file.exists()

    class DescribeReadJoblib:
        """Tests for the read_joblib method."""

        def it_reads_joblib_file(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify reads joblib file."""
            joblib_file = tmp_path / "data.joblib"
            obj = {"key": "value", "array": [1, 2, 3]}
            joblib.dump(obj, joblib_file)

            loaded = storage.read_joblib(f"file://{joblib_file}")

            assert loaded == obj

        def it_supports_mmap_mode(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify supports memory-mapping mode."""
            import numpy as np

            joblib_file = tmp_path / "array.joblib"
            arr = np.array([1, 2, 3, 4, 5])
            joblib.dump(arr, joblib_file)

            loaded = storage.read_joblib(f"file://{joblib_file}", mmap_mode="r")

            assert np.array_equal(loaded, arr)

        def it_raises_for_nonexistent_file(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify raises FileDoesNotExistError for missing file."""
            nonexistent = tmp_path / "nonexistent.joblib"

            with pytest.raises(FileDoesNotExistError):
                storage.read_joblib(f"file://{nonexistent}")

    class DescribeWriteJoblib:
        """Tests for the write_joblib method."""

        def it_writes_object_to_joblib(self, storage: LocalStorageService, tmp_path: Path) -> None:
            """Verify writes object to joblib file."""
            joblib_file = tmp_path / "data.joblib"
            obj = {"key": "value", "array": [1, 2, 3]}

            storage.write_joblib(obj, f"file://{joblib_file}")

            assert joblib_file.exists()
            assert joblib.load(joblib_file) == obj

        def it_creates_parent_directories(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify creates parent directories if needed."""
            joblib_file = tmp_path / "nested" / "data.joblib"
            obj = {"key": "value"}

            storage.write_joblib(obj, f"file://{joblib_file}")

            assert joblib_file.exists()

    class DescribeLocalCache:
        """Tests for the local_cache context manager."""

        def it_yields_path_directly_for_local_files(
            self, storage: LocalStorageService, tmp_path: Path
        ) -> None:
            """Verify yields the path directly for local storage."""
            test_file = tmp_path / "test.txt"
            test_file.write_text("content")

            with storage.local_cache(f"file://{test_file}") as cached_path:
                assert cached_path == test_file


class DescribeStorageErrors:
    """Tests for storage exception classes."""

    def it_creates_storage_error_with_message(self) -> None:
        """Verify StorageError contains URI and reason."""
        error = StorageError("s3://bucket/file.txt", "Access denied")

        assert "s3://bucket/file.txt" in str(error)
        assert "Access denied" in str(error)
        assert error.uri == "s3://bucket/file.txt"
        assert error.reason == "Access denied"

    def it_creates_file_does_not_exist_error(self) -> None:
        """Verify FileDoesNotExistError contains URI."""
        error = FileDoesNotExistError("file:///tmp/missing.txt")

        assert "file:///tmp/missing.txt" in str(error)
        assert "File does not exist" in str(error)
