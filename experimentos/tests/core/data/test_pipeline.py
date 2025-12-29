"""Tests for the data processing pipeline module."""

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.core.data.pipeline import (
    DataProcessingPipeline,
    DataProcessingPipelineFactory,
)
from experiments.services.storage import StorageService
from experiments.services.storage.local import LocalStorageService


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )


@pytest.fixture
def mock_loader(sample_dataframe: pl.DataFrame) -> MagicMock:
    """Create a mock loader."""
    loader = MagicMock()
    loader.load.return_value = sample_dataframe
    return loader


@pytest.fixture
def mock_transformer(sample_dataframe: pl.DataFrame) -> MagicMock:
    """Create a mock transformer."""
    transformer = MagicMock()
    transformer.transform.return_value = sample_dataframe
    return transformer


@pytest.fixture
def mock_exporter(tmp_path: Path) -> MagicMock:
    """Create a mock exporter."""
    exporter = MagicMock()
    output_uri = str(tmp_path / "output.parquet")
    exporter.export.return_value = output_uri
    return exporter


@pytest.fixture
def storage() -> StorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


class DescribeDataProcessingPipeline:
    """Tests for DataProcessingPipeline."""

    def it_orchestrates_load_transform_export(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Verify pipeline calls loader, transformer, and exporter in order."""
        pipeline = DataProcessingPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
        )

        pipeline.run(Dataset.TAIWAN_CREDIT)

        # Verify all stages were called
        mock_loader.load.assert_called_once_with(Dataset.TAIWAN_CREDIT)
        mock_transformer.transform.assert_called_once()
        mock_exporter.export.assert_called_once()

    def it_passes_loaded_data_to_transformer(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Verify loader output is passed to transformer."""
        pipeline = DataProcessingPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
        )

        pipeline.run(Dataset.TAIWAN_CREDIT)

        # First arg to transform should be the loaded dataframe
        call_args = mock_transformer.transform.call_args
        passed_df = call_args[0][0]
        assert passed_df.shape == sample_dataframe.shape

    def it_passes_transformed_data_to_exporter(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        sample_dataframe: pl.DataFrame,
    ) -> None:
        """Verify transformer output is passed to exporter."""
        pipeline = DataProcessingPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
        )

        pipeline.run(Dataset.TAIWAN_CREDIT)

        # First arg to export should be the transformed dataframe
        call_args = mock_exporter.export.call_args
        passed_df = call_args[0][0]
        assert passed_df.shape == sample_dataframe.shape

    def it_returns_exported_path(
        self,
        mock_loader: MagicMock,
        mock_transformer: MagicMock,
        mock_exporter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify pipeline returns the path from exporter."""
        expected_path = tmp_path / "output.parquet"
        mock_exporter.export.return_value = expected_path

        pipeline = DataProcessingPipeline(
            loader=mock_loader,
            transformer=mock_transformer,
            exporter=mock_exporter,
        )

        result = pipeline.run(Dataset.TAIWAN_CREDIT)

        assert result == expected_path


class DescribeDataProcessingPipelineFactory:
    """Tests for DataProcessingPipelineFactory."""

    def it_creates_pipeline_for_taiwan_credit(
        self, tmp_path: Path, storage: StorageService
    ) -> None:
        """Verify factory creates pipeline for Taiwan Credit dataset."""
        factory = DataProcessingPipelineFactory(
            storage=storage,
            raw_data_uri=str(tmp_path / "raw"),
            interim_data_uri=str(tmp_path / "interim"),
        )

        pipeline = factory.create(Dataset.TAIWAN_CREDIT)

        assert isinstance(pipeline, DataProcessingPipeline)

    def it_creates_pipeline_for_lending_club(
        self, tmp_path: Path, storage: StorageService
    ) -> None:
        """Verify factory creates pipeline for Lending Club dataset."""
        factory = DataProcessingPipelineFactory(
            storage=storage,
            raw_data_uri=str(tmp_path / "raw"),
            interim_data_uri=str(tmp_path / "interim"),
        )

        pipeline = factory.create(Dataset.LENDING_CLUB)

        assert isinstance(pipeline, DataProcessingPipeline)

    def it_creates_pipeline_for_corporate_credit(
        self, tmp_path: Path, storage: StorageService
    ) -> None:
        """Verify factory creates pipeline for Corporate Credit dataset."""
        factory = DataProcessingPipelineFactory(
            storage=storage,
            raw_data_uri=str(tmp_path / "raw"),
            interim_data_uri=str(tmp_path / "interim"),
        )

        pipeline = factory.create(Dataset.CORPORATE_CREDIT_RATING)

        assert isinstance(pipeline, DataProcessingPipeline)

    def it_passes_use_gpu_to_transformer(self, tmp_path: Path, storage: StorageService) -> None:
        """Verify GPU flag is passed to transformer."""
        factory = DataProcessingPipelineFactory(
            storage=storage,
            raw_data_uri=str(tmp_path / "raw"),
            interim_data_uri=str(tmp_path / "interim"),
            use_gpu=True,
        )

        pipeline = factory.create(Dataset.TAIWAN_CREDIT)

        # Access the transformer through the pipeline's private attribute
        assert pipeline._transformer.use_gpu is True

    def it_raises_for_unknown_dataset(self, tmp_path: Path, storage: StorageService) -> None:
        """Verify factory raises ValueError for unregistered datasets."""
        factory = DataProcessingPipelineFactory(
            storage=storage,
            raw_data_uri=str(tmp_path / "raw"),
            interim_data_uri=str(tmp_path / "interim"),
        )

        # Create a mock dataset that's not in the registry
        mock_dataset = MagicMock()
        mock_dataset.id = "unknown_dataset"

        with pytest.raises(ValueError, match="No transformer registered"):
            factory.create(mock_dataset)
