"""Tests for the data loaders module."""

from pathlib import Path

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.core.data.loaders import CsvRawDataLoader
from experiments.core.data.protocols import RawDataUriProvider
from experiments.services.storage import (
    FileDoesNotExistError,
    LocalStorageService,
    StorageService,
)


class FakeUriProvider:
    """Fake implementation of RawDataUriProvider for testing."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def get_raw_data_uri(self, dataset_id: str) -> str:
        if self._path is None:
            return f"file:///mock/raw/{dataset_id}.csv"
        return StorageService.to_uri(self._path)


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
    def storage(self) -> LocalStorageService:
        """Create a local storage service."""
        return LocalStorageService()

    def it_satisfies_raw_data_uri_provider_protocol(self) -> None:
        """Ensure FakeUriProvider satisfies RawDataUriProvider protocol check."""
        provider = FakeUriProvider()
        assert isinstance(provider, RawDataUriProvider)

    def it_loads_dataframe_from_csv(
        self, tmp_path: Path, sample_csv_content: str, storage: LocalStorageService
    ) -> None:
        """Verify loader reads CSV and returns a Polars DataFrame."""
        # Create file with correct name for the dataset
        csv_path = tmp_path / "taiwan_credit.csv"
        csv_path.write_text(sample_csv_content)

        base_uri = StorageService.to_uri(tmp_path)
        loader = CsvRawDataLoader(storage, base_uri)

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "loan_amnt" in result.columns
        assert "target" in result.columns

    def it_returns_correct_column_types(
        self, tmp_path: Path, sample_csv_content: str, storage: LocalStorageService
    ) -> None:
        """Verify loader preserves numeric types."""
        # Create file with correct name
        csv_path = tmp_path / "taiwan_credit.csv"
        csv_path.write_text(sample_csv_content)

        base_uri = StorageService.to_uri(tmp_path)
        loader = CsvRawDataLoader(storage, base_uri)

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result["loan_amnt"].dtype == pl.Int64
        assert result["target"].dtype == pl.Int64

    def it_raises_error_when_file_not_found(
        self, tmp_path: Path, storage: LocalStorageService
    ) -> None:
        """Verify loader raises FileDoesNotExistError for missing files."""
        base_uri = StorageService.to_uri(tmp_path)
        loader = CsvRawDataLoader(storage, base_uri)

        with pytest.raises(FileDoesNotExistError):
            loader.load(Dataset.TAIWAN_CREDIT)

    def it_applies_dataset_extra_params(
        self, tmp_path: Path, storage: LocalStorageService
    ) -> None:
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

        base_uri = StorageService.to_uri(tmp_path)
        loader = CsvRawDataLoader(storage, base_uri)

        result = loader.load(Dataset.LENDING_CLUB)

        # The id column should be string due to schema_overrides
        assert result["id"].dtype == pl.Utf8
        assert result["id"].to_list() == ["abc123", "xyz789"]
