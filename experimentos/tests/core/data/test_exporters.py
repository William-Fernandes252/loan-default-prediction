"""Tests for the data exporters module."""

from pathlib import Path

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.core.data.exporters import ParquetDataExporter
from experiments.core.data.protocols import InterimDataPathProvider, ProcessedDataExporter


class FakeInterimPathProvider:
    """Fake implementation of InterimDataPathProvider for testing."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def get_interim_data_path(self, dataset_id: str) -> Path:
        return self._base_path / f"{dataset_id}.parquet"


class DescribeParquetDataExporter:
    """Tests for ParquetDataExporter."""

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

    def it_satisfies_processed_data_exporter_protocol(self, tmp_path: Path) -> None:
        """Verify ParquetDataExporter satisfies the protocol."""
        exporter = ParquetDataExporter(FakeInterimPathProvider(tmp_path))
        assert isinstance(exporter, ProcessedDataExporter)

    def it_satisfies_interim_data_path_provider_protocol(self, tmp_path: Path) -> None:
        """Verify FakeInterimPathProvider satisfies the protocol."""
        provider = FakeInterimPathProvider(tmp_path)
        assert isinstance(provider, InterimDataPathProvider)

    def it_exports_dataframe_to_parquet(
        self, tmp_path: Path, sample_dataframe: pl.DataFrame
    ) -> None:
        """Verify exporter writes DataFrame to Parquet file."""
        exporter = ParquetDataExporter(FakeInterimPathProvider(tmp_path))

        output_path = exporter.export(sample_dataframe, Dataset.TAIWAN_CREDIT)

        assert output_path.exists()
        assert output_path.suffix == ".parquet"

    def it_returns_correct_output_path(
        self, tmp_path: Path, sample_dataframe: pl.DataFrame
    ) -> None:
        """Verify returned path matches expected location."""
        exporter = ParquetDataExporter(FakeInterimPathProvider(tmp_path))

        output_path = exporter.export(sample_dataframe, Dataset.TAIWAN_CREDIT)

        expected_path = tmp_path / "taiwan_credit.parquet"
        assert output_path == expected_path

    def it_creates_parent_directories(
        self, tmp_path: Path, sample_dataframe: pl.DataFrame
    ) -> None:
        """Verify exporter creates parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "dirs"
        exporter = ParquetDataExporter(FakeInterimPathProvider(nested_path))

        output_path = exporter.export(sample_dataframe, Dataset.LENDING_CLUB)

        assert output_path.parent.exists()
        assert output_path.exists()

    def it_preserves_dataframe_content(
        self, tmp_path: Path, sample_dataframe: pl.DataFrame
    ) -> None:
        """Verify exported Parquet contains the same data."""
        exporter = ParquetDataExporter(FakeInterimPathProvider(tmp_path))

        output_path = exporter.export(sample_dataframe, Dataset.TAIWAN_CREDIT)

        # Read back the exported file
        loaded_df = pl.read_parquet(output_path)

        assert loaded_df.shape == sample_dataframe.shape
        assert loaded_df.columns == sample_dataframe.columns
        assert loaded_df["feature1"].to_list() == [1, 2, 3]
        assert loaded_df["target"].to_list() == [0, 1, 0]

    def it_overwrites_existing_file(self, tmp_path: Path, sample_dataframe: pl.DataFrame) -> None:
        """Verify exporter overwrites existing file without error."""
        exporter = ParquetDataExporter(FakeInterimPathProvider(tmp_path))

        # Export first version
        output_path = exporter.export(sample_dataframe, Dataset.TAIWAN_CREDIT)

        # Export second version with different data
        new_df = pl.DataFrame({"new_col": [10, 20]})
        output_path_2 = exporter.export(new_df, Dataset.TAIWAN_CREDIT)

        assert output_path == output_path_2

        # Verify new data is there
        loaded_df = pl.read_parquet(output_path)
        assert "new_col" in loaded_df.columns
        assert "feature1" not in loaded_df.columns
