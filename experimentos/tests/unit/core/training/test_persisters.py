"""Tests for experiments.core.training.persisters module."""

from unittest.mock import MagicMock

import pytest

from experiments.core.data import Dataset
from experiments.core.training.persisters import (
    ConsolidationUriProvider,
    ParquetCheckpointPersister,
)
from experiments.services.storage import StorageService
from experiments.services.storage.local import LocalStorageService


@pytest.fixture
def storage() -> StorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


class DescribeConsolidationUriProvider:
    """Tests for ConsolidationUriProvider protocol."""

    def it_defines_get_checkpoint_uri_method(self) -> None:
        """Verify protocol defines get_checkpoint_uri method."""

        class ValidProvider:
            def get_checkpoint_uri(
                self,
                dataset_id: str,
                model_id: str,
                technique_id: str,
                seed: int,
            ) -> str:
                return "/checkpoints/test.parquet"

            def get_consolidated_results_uri(self, dataset_id: str) -> str:
                return "/output.parquet"

        # Class exists and has required methods
        assert hasattr(ValidProvider, "get_checkpoint_uri")
        assert hasattr(ValidProvider, "get_consolidated_results_uri")

    def it_defines_get_consolidated_results_uri_method(self) -> None:
        """Verify protocol defines get_consolidated_results_uri method."""
        # Check protocol has the method
        assert hasattr(ConsolidationUriProvider, "get_consolidated_results_uri")

    def it_is_runtime_checkable(self) -> None:
        """Verify ConsolidationUriProvider can be checked at runtime."""

        class ValidProvider:
            def get_checkpoint_uri(
                self,
                dataset_id: str,
                model_id: str,
                technique_id: str,
                seed: int,
            ) -> str:
                return "/checkpoints/test.parquet"

            def get_consolidated_results_uri(self, dataset_id: str) -> str:
                return "/output.parquet"

        provider = ValidProvider()
        assert isinstance(provider, ConsolidationUriProvider)


class DescribeParquetCheckpointPersister:
    """Tests for ParquetCheckpointPersister class."""

    def it_initializes_with_storage_and_providers(self, storage: StorageService) -> None:
        """Verify initialization stores storage and providers."""
        mock_checkpoint_provider = MagicMock()
        mock_results_provider = MagicMock()
        persister = ParquetCheckpointPersister(
            storage=storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        assert persister._storage is storage
        assert persister._checkpoint_uri_provider is mock_checkpoint_provider
        assert persister._results_uri_provider is mock_results_provider


class DescribeParquetCheckpointPersisterGetCheckpointDir:
    """Tests for ParquetCheckpointPersister._get_checkpoint_dir_uri() method."""

    def it_uses_uri_provider_to_get_directory(self, storage: StorageService) -> None:
        """Verify checkpoint directory is derived from URI provider."""
        mock_checkpoint_provider = MagicMock()
        mock_checkpoint_provider.get_checkpoint_uri.return_value = (
            "/results/taiwan_credit/checkpoints/test.parquet"
        )
        mock_results_provider = MagicMock()
        persister = ParquetCheckpointPersister(
            storage=storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        result = persister._get_checkpoint_dir_uri(Dataset.TAIWAN_CREDIT)

        mock_checkpoint_provider.get_checkpoint_uri.assert_called_once_with(
            Dataset.TAIWAN_CREDIT.id, "x", "y", 0
        )
        assert result == "/results/taiwan_credit/checkpoints"


class DescribeParquetCheckpointPersisterConsolidate:
    """Tests for ParquetCheckpointPersister.consolidate() method."""

    @pytest.fixture
    def mock_checkpoint_provider(self) -> MagicMock:
        """Create mock checkpoint URI provider."""
        provider = MagicMock()
        provider.get_checkpoint_uri.return_value = "/checkpoints/test/sample.parquet"
        return provider

    @pytest.fixture
    def mock_results_provider(self) -> MagicMock:
        """Create mock results URI provider."""
        provider = MagicMock()
        provider.get_consolidated_results_uri.return_value = "/output/results.parquet"
        return provider

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create mock storage service."""
        return MagicMock(spec=StorageService)

    def it_returns_none_when_no_checkpoint_files(
        self,
        mock_storage: MagicMock,
        mock_checkpoint_provider: MagicMock,
        mock_results_provider: MagicMock,
    ) -> None:
        """Verify None is returned when no checkpoints exist."""
        mock_storage.list_files.return_value = []
        persister = ParquetCheckpointPersister(
            storage=mock_storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        result = persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert result is None

    def it_reads_all_parquet_files(
        self,
        mock_storage: MagicMock,
        mock_checkpoint_provider: MagicMock,
        mock_results_provider: MagicMock,
    ) -> None:
        """Verify all parquet files are read."""
        import polars as pl

        checkpoint_uris = [
            "/checkpoints/test/0.parquet",
            "/checkpoints/test/1.parquet",
        ]
        sample_frames = [
            pl.DataFrame({"accuracy": [0.85], "seed": [0]}),
            pl.DataFrame({"accuracy": [0.87], "seed": [1]}),
        ]
        mock_storage.list_files.return_value = checkpoint_uris
        mock_storage.read_parquet.side_effect = sample_frames

        persister = ParquetCheckpointPersister(
            storage=mock_storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert mock_storage.read_parquet.call_count == 2

    def it_saves_consolidated_dataframe_using_storage(
        self,
        mock_storage: MagicMock,
        mock_checkpoint_provider: MagicMock,
        mock_results_provider: MagicMock,
    ) -> None:
        """Verify consolidated dataframe is saved via storage service."""
        import polars as pl

        checkpoint_uris = ["/checkpoints/test/0.parquet"]
        sample_frame = pl.DataFrame({"accuracy": [0.85], "seed": [0]})
        mock_storage.list_files.return_value = checkpoint_uris
        mock_storage.read_parquet.return_value = sample_frame

        persister = ParquetCheckpointPersister(
            storage=mock_storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        persister.consolidate(Dataset.TAIWAN_CREDIT)

        mock_storage.write_parquet.assert_called_once()

    def it_gets_consolidated_results_uri_from_provider(
        self,
        mock_storage: MagicMock,
        mock_checkpoint_provider: MagicMock,
        mock_results_provider: MagicMock,
    ) -> None:
        """Verify consolidated results URI is retrieved from provider."""
        import polars as pl

        checkpoint_uris = ["/checkpoints/test/0.parquet"]
        sample_frame = pl.DataFrame({"accuracy": [0.85], "seed": [0]})
        mock_storage.list_files.return_value = checkpoint_uris
        mock_storage.read_parquet.return_value = sample_frame

        persister = ParquetCheckpointPersister(
            storage=mock_storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        persister.consolidate(Dataset.TAIWAN_CREDIT)

        mock_results_provider.get_consolidated_results_uri.assert_called_once_with(
            Dataset.TAIWAN_CREDIT.id
        )

    def it_returns_output_uri_on_success(
        self,
        mock_storage: MagicMock,
        mock_checkpoint_provider: MagicMock,
        mock_results_provider: MagicMock,
    ) -> None:
        """Verify output URI is returned on success."""
        import polars as pl

        mock_results_provider.get_consolidated_results_uri.return_value = "/output/results.parquet"
        checkpoint_uris = ["/checkpoints/test/0.parquet"]
        sample_frame = pl.DataFrame({"accuracy": [0.85], "seed": [0]})
        mock_storage.list_files.return_value = checkpoint_uris
        mock_storage.read_parquet.return_value = sample_frame

        persister = ParquetCheckpointPersister(
            storage=mock_storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        result = persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert result == "/output/results.parquet"

    def it_returns_none_when_no_valid_frames_loaded(
        self,
        mock_storage: MagicMock,
        mock_checkpoint_provider: MagicMock,
        mock_results_provider: MagicMock,
    ) -> None:
        """Verify None is returned when all reads fail."""
        checkpoint_uris = ["/checkpoints/test/0.parquet"]
        mock_storage.list_files.return_value = checkpoint_uris
        mock_storage.read_parquet.side_effect = Exception("Read error")

        persister = ParquetCheckpointPersister(
            storage=mock_storage,
            checkpoint_uri_provider=mock_checkpoint_provider,
            results_uri_provider=mock_results_provider,
        )

        result = persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert result is None
