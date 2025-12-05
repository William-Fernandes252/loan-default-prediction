"""Tests for the data protocols module."""

from pathlib import Path

import polars as pl

from experiments.core.data import Dataset
from experiments.core.data.protocols import (
    DataTransformer,
    InterimDataPathProvider,
    ProcessedDataExporter,
    RawDataLoader,
    RawDataPathProvider,
)


class FakeRawDataPathProvider:
    """Fake implementation of RawDataPathProvider for testing."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def get_raw_data_path(self, dataset_id: str) -> Path:
        return self._base_path / f"{dataset_id}.csv"


class FakeInterimDataPathProvider:
    """Fake implementation of InterimDataPathProvider for testing."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def get_interim_data_path(self, dataset_id: str) -> Path:
        return self._base_path / f"{dataset_id}.parquet"


class FakeRawDataLoader:
    """Fake implementation of RawDataLoader for testing."""

    def __init__(self, data: pl.DataFrame | None = None) -> None:
        self._data = data if data is not None else pl.DataFrame()

    def load(self, dataset: Dataset) -> pl.DataFrame:
        return self._data


class FakeDataTransformer:
    """Fake implementation of DataTransformer for testing."""

    def __init__(self, transform_fn=None) -> None:
        self._transform_fn = transform_fn or (lambda df, ds: df)

    def transform(self, df: pl.DataFrame, dataset: Dataset) -> pl.DataFrame:
        return self._transform_fn(df, dataset)


class FakeProcessedDataExporter:
    """Fake implementation of ProcessedDataExporter for testing."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self.exported_data: pl.DataFrame | None = None

    def export(self, df: pl.DataFrame, dataset: Dataset) -> Path:
        self.exported_data = df
        return self._output_path


class DescribeRawDataPathProvider:
    """Tests for the RawDataPathProvider protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        provider = FakeRawDataPathProvider(Path("/base"))
        assert isinstance(provider, RawDataPathProvider)

    def it_returns_correct_path_for_dataset(self) -> None:
        """Verify path is constructed correctly."""
        provider = FakeRawDataPathProvider(Path("/data/raw"))
        path = provider.get_raw_data_path("taiwan_credit")
        assert path == Path("/data/raw/taiwan_credit.csv")


class DescribeInterimDataPathProvider:
    """Tests for the InterimDataPathProvider protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        provider = FakeInterimDataPathProvider(Path("/base"))
        assert isinstance(provider, InterimDataPathProvider)

    def it_returns_correct_path_for_dataset(self) -> None:
        """Verify path is constructed correctly."""
        provider = FakeInterimDataPathProvider(Path("/data/interim"))
        path = provider.get_interim_data_path("lending_club")
        assert path == Path("/data/interim/lending_club.parquet")


class DescribeRawDataLoader:
    """Tests for the RawDataLoader protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        loader = FakeRawDataLoader()
        assert isinstance(loader, RawDataLoader)

    def it_loads_data_for_dataset(self) -> None:
        """Verify loader returns data for a dataset."""
        sample_df = pl.DataFrame({"col": [1, 2, 3]})
        loader = FakeRawDataLoader(sample_df)

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.shape == (3, 1)
        assert result["col"].to_list() == [1, 2, 3]


class DescribeDataTransformer:
    """Tests for the DataTransformer protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        transformer = FakeDataTransformer()
        assert isinstance(transformer, DataTransformer)

    def it_transforms_dataframe(self) -> None:
        """Verify transformer applies transformation."""
        sample_df = pl.DataFrame({"value": [1, 2, 3]})

        def double_values(df: pl.DataFrame, dataset: Dataset) -> pl.DataFrame:
            return df.with_columns(pl.col("value") * 2)

        transformer = FakeDataTransformer(double_values)
        result = transformer.transform(sample_df, Dataset.TAIWAN_CREDIT)

        assert result["value"].to_list() == [2, 4, 6]


class DescribeProcessedDataExporter:
    """Tests for the ProcessedDataExporter protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        exporter = FakeProcessedDataExporter(Path("/output/data.parquet"))
        assert isinstance(exporter, ProcessedDataExporter)

    def it_exports_dataframe_and_returns_path(self) -> None:
        """Verify exporter stores data and returns output path."""
        output_path = Path("/output/taiwan_credit.parquet")
        exporter = FakeProcessedDataExporter(output_path)
        sample_df = pl.DataFrame({"col": [1, 2, 3]})

        result_path = exporter.export(sample_df, Dataset.TAIWAN_CREDIT)

        assert result_path == output_path
        assert exporter.exported_data is not None
        assert exporter.exported_data.shape == (3, 1)
