"""Tests for the data protocols module."""

import polars as pl

from experiments.core.data import Dataset
from experiments.core.data.protocols import (
    DataTransformer,
    InterimDataUriProvider,
    ProcessedDataExporter,
    RawDataLoader,
    RawDataUriProvider,
)


class FakeRawDataUriProvider:
    """Fake implementation of RawDataUriProvider for testing."""

    def __init__(self, base_uri: str) -> None:
        self._base_uri = base_uri.rstrip("/")

    def get_raw_data_uri(self, dataset_id: str) -> str:
        return f"{self._base_uri}/{dataset_id}.csv"


class FakeInterimDataUriProvider:
    """Fake implementation of InterimDataUriProvider for testing."""

    def __init__(self, base_uri: str) -> None:
        self._base_uri = base_uri.rstrip("/")

    def get_interim_data_uri(self, dataset_id: str) -> str:
        return f"{self._base_uri}/{dataset_id}.parquet"


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

    def __init__(self, output_uri: str) -> None:
        self._output_uri = output_uri
        self.exported_data: pl.DataFrame | None = None

    def export(self, df: pl.DataFrame, dataset: Dataset) -> str:
        self.exported_data = df
        return self._output_uri


class DescribeRawDataUriProvider:
    """Tests for the RawDataUriProvider protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        provider = FakeRawDataUriProvider("/base")
        assert isinstance(provider, RawDataUriProvider)

    def it_returns_correct_uri_for_dataset(self) -> None:
        """Verify URI is constructed correctly."""
        provider = FakeRawDataUriProvider("/data/raw")
        uri = provider.get_raw_data_uri("taiwan_credit")
        assert uri == "/data/raw/taiwan_credit.csv"


class DescribeInterimDataUriProvider:
    """Tests for the InterimDataUriProvider protocol."""

    def it_satisfies_protocol_with_fake_implementation(self) -> None:
        """Verify fake implementation satisfies the protocol."""
        provider = FakeInterimDataUriProvider("/base")
        assert isinstance(provider, InterimDataUriProvider)

    def it_returns_correct_uri_for_dataset(self) -> None:
        """Verify URI is constructed correctly."""
        provider = FakeInterimDataUriProvider("/data/interim")
        uri = provider.get_interim_data_uri("lending_club")
        assert uri == "/data/interim/lending_club.parquet"


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
        exporter = FakeProcessedDataExporter("/output/data.parquet")
        assert isinstance(exporter, ProcessedDataExporter)

    def it_exports_dataframe_and_returns_uri(self) -> None:
        """Verify exporter stores data and returns output URI."""
        output_uri = "/output/taiwan_credit.parquet"
        exporter = FakeProcessedDataExporter(output_uri)
        sample_df = pl.DataFrame({"col": [1, 2, 3]})

        result_uri = exporter.export(sample_df, Dataset.TAIWAN_CREDIT)

        assert result_uri == output_uri
        assert exporter.exported_data is not None
        assert exporter.exported_data.shape == (3, 1)
