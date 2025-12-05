"""Tests for the data loaders module."""

from pathlib import Path

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.core.data.loaders import CsvRawDataLoader
from experiments.core.data.protocols import RawDataPathProvider


class FakePathProvider:
    """Fake implementation of RawDataPathProvider for testing."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def get_raw_data_path(self, dataset_id: str) -> Path:
        if self._path is None:
            return Path(f"/mock/raw/{dataset_id}.csv")
        return self._path


class DescribeCsvRawDataLoader:
    """Tests for CsvRawDataLoader."""

    @pytest.fixture
    def sample_csv_content(self) -> str:
        """Create sample CSV content."""
        return """id,loan_amnt,target
1,5000,0
2,10000,1
3,15000,0
"""

    @pytest.fixture
    def csv_path(self, tmp_path: Path, sample_csv_content: str) -> Path:
        """Create a temporary CSV file with sample data."""
        path = tmp_path / "test_data.csv"
        path.write_text(sample_csv_content)
        return path

    def it_satisfies_raw_data_loader_protocol(self) -> None:
        """Ensure CsvRawDataLoader satisfies RawDataPathProvider protocol check."""
        provider = FakePathProvider()
        assert isinstance(provider, RawDataPathProvider)

    def it_loads_dataframe_from_csv(self, csv_path: Path) -> None:
        """Verify loader reads CSV and returns a Polars DataFrame."""
        loader = CsvRawDataLoader(FakePathProvider(csv_path))

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "loan_amnt" in result.columns
        assert "target" in result.columns

    def it_returns_correct_column_types(self, csv_path: Path) -> None:
        """Verify loader preserves numeric types."""
        loader = CsvRawDataLoader(FakePathProvider(csv_path))

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result["loan_amnt"].dtype == pl.Int64
        assert result["target"].dtype == pl.Int64

    def it_raises_error_when_file_not_found(self, tmp_path: Path) -> None:
        """Verify loader raises FileNotFoundError for missing files."""
        nonexistent_path = tmp_path / "nonexistent.csv"
        loader = CsvRawDataLoader(FakePathProvider(nonexistent_path))

        with pytest.raises(FileNotFoundError):
            loader.load(Dataset.TAIWAN_CREDIT)

    def it_applies_dataset_extra_params(self, tmp_path: Path) -> None:
        """Verify loader applies dataset-specific parameters.

        For LENDING_CLUB, the 'id' column should be read as string due to
        schema_overrides in get_extra_params().
        """
        # Create a CSV with id column that would normally be parsed as int
        csv_content = """id,loan_amnt
abc123,5000
xyz789,10000
"""
        csv_path = tmp_path / "lending_club.csv"
        csv_path.write_text(csv_content)

        loader = CsvRawDataLoader(FakePathProvider(csv_path))
        result = loader.load(Dataset.LENDING_CLUB)

        # The id column should be string due to schema_overrides
        assert result["id"].dtype == pl.Utf8
        assert result["id"].to_list() == ["abc123", "xyz789"]
