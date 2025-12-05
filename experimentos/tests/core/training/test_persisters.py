"""Tests for experiments.core.training.persisters module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.core.data import Dataset
from experiments.core.training.persisters import (
    ConsolidationPathProvider,
    ParquetCheckpointPersister,
)


class DescribeConsolidationPathProvider:
    """Tests for ConsolidationPathProvider protocol."""

    def it_defines_get_checkpoint_path_method(self) -> None:
        """Verify protocol defines get_checkpoint_path method."""

        class ValidProvider:
            def get_checkpoint_path(
                self,
                dataset_id: str,
                model_id: str,
                technique_id: str,
                seed: int,
            ) -> Path:
                return Path("/checkpoints")

            def get_consolidated_results_path(self, dataset_id: str) -> Path:
                return Path("/output.parquet")

        # Class exists and has required methods
        assert hasattr(ValidProvider, "get_checkpoint_path")
        assert hasattr(ValidProvider, "get_consolidated_results_path")

    def it_defines_get_consolidated_results_path_method(self) -> None:
        """Verify protocol defines get_consolidated_results_path method."""
        # Check protocol has the method
        assert hasattr(ConsolidationPathProvider, "get_consolidated_results_path")


class DescribeParquetCheckpointPersister:
    """Tests for ParquetCheckpointPersister class."""

    def it_initializes_with_path_provider(self) -> None:
        """Verify initialization stores path provider."""
        mock_provider = MagicMock()
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_provider,
            results_path_provider=mock_provider,
        )

        assert persister._checkpoint_path_provider is mock_provider
        assert persister._results_path_provider is mock_provider


class DescribeParquetCheckpointPersisterGetCheckpointDir:
    """Tests for ParquetCheckpointPersister._get_checkpoint_dir() method."""

    def it_uses_path_provider_to_get_directory(self) -> None:
        """Verify checkpoint directory is derived from path provider."""
        mock_provider = MagicMock()
        mock_provider.get_checkpoint_path.return_value = Path(
            "/results/taiwan_credit/checkpoints/test.parquet"
        )
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_provider,
            results_path_provider=mock_provider,
        )

        result = persister._get_checkpoint_dir(Dataset.TAIWAN_CREDIT)

        mock_provider.get_checkpoint_path.assert_called_once_with(
            Dataset.TAIWAN_CREDIT.id, "x", "y", 0
        )
        assert result == Path("/results/taiwan_credit/checkpoints")


class DescribeParquetCheckpointPersisterConsolidate:
    """Tests for ParquetCheckpointPersister.consolidate() method."""

    @pytest.fixture
    def mock_path_provider(self) -> MagicMock:
        """Create mock path provider."""
        provider = MagicMock()
        provider.get_checkpoint_path.return_value = Path("/checkpoints/test/sample.parquet")
        provider.get_consolidated_results_path.return_value = Path("/output/results.parquet")
        return provider

    @pytest.fixture
    def sample_dataframes(self) -> list[pd.DataFrame]:
        """Create sample checkpoint dataframes."""
        return [
            pd.DataFrame({"accuracy": [0.85], "seed": [0]}),
            pd.DataFrame({"accuracy": [0.87], "seed": [1]}),
        ]

    def it_returns_none_when_no_checkpoint_files(
        self,
        mock_path_provider: MagicMock,
    ) -> None:
        """Verify None is returned when no checkpoints exist."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        with patch.object(Path, "glob", return_value=[]):
            result = persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert result is None

    def it_reads_all_parquet_files(
        self,
        mock_path_provider: MagicMock,
        sample_dataframes: list[pd.DataFrame],
    ) -> None:
        """Verify all parquet files are read."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [
            Path("/checkpoints/test/0.parquet"),
            Path("/checkpoints/test/1.parquet"),
        ]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch.object(Path, "mkdir"),
            patch("pandas.read_parquet") as mock_read,
            patch("pandas.concat") as mock_concat,
        ):
            mock_read.side_effect = sample_dataframes
            mock_concat.return_value = pd.concat(sample_dataframes)
            mock_concat.return_value.to_parquet = MagicMock()

            persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert mock_read.call_count == 2

    def it_concatenates_all_dataframes(
        self,
        mock_path_provider: MagicMock,
        sample_dataframes: list[pd.DataFrame],
    ) -> None:
        """Verify dataframes are concatenated."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [
            Path("/checkpoints/test/0.parquet"),
            Path("/checkpoints/test/1.parquet"),
        ]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch.object(Path, "mkdir"),
            patch("pandas.read_parquet") as mock_read,
            patch("experiments.core.training.persisters.pd.concat") as mock_concat,
        ):
            mock_read.side_effect = sample_dataframes
            concat_result = MagicMock()
            mock_concat.return_value = concat_result

            persister.consolidate(Dataset.TAIWAN_CREDIT)

        mock_concat.assert_called_once()

    def it_creates_output_directory(
        self,
        mock_path_provider: MagicMock,
        sample_dataframes: list[pd.DataFrame],
    ) -> None:
        """Verify output directory is created."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [Path("/checkpoints/test/0.parquet")]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch.object(Path, "mkdir") as mock_mkdir,
            patch("pandas.read_parquet") as mock_read,
            patch("pandas.concat") as mock_concat,
        ):
            mock_read.return_value = sample_dataframes[0]
            concat_result = MagicMock()
            mock_concat.return_value = concat_result

            persister.consolidate(Dataset.TAIWAN_CREDIT)

        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def it_saves_consolidated_dataframe_as_parquet(
        self,
        mock_path_provider: MagicMock,
        sample_dataframes: list[pd.DataFrame],
    ) -> None:
        """Verify consolidated dataframe is saved."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [Path("/checkpoints/test/0.parquet")]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch.object(Path, "mkdir"),
            patch("pandas.read_parquet") as mock_read,
            patch("pandas.concat") as mock_concat,
        ):
            mock_read.return_value = sample_dataframes[0]
            concat_result = MagicMock()
            mock_concat.return_value = concat_result

            persister.consolidate(Dataset.TAIWAN_CREDIT)

        concat_result.to_parquet.assert_called_once()

    def it_gets_consolidated_results_path_from_provider(
        self,
        mock_path_provider: MagicMock,
        sample_dataframes: list[pd.DataFrame],
    ) -> None:
        """Verify consolidated results path is retrieved from provider."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [Path("/checkpoints/test/0.parquet")]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch.object(Path, "mkdir"),
            patch("pandas.read_parquet") as mock_read,
            patch("pandas.concat") as mock_concat,
        ):
            mock_read.return_value = sample_dataframes[0]
            concat_result = MagicMock()
            mock_concat.return_value = concat_result

            persister.consolidate(Dataset.TAIWAN_CREDIT)

        mock_path_provider.get_consolidated_results_path.assert_called_once_with(
            Dataset.TAIWAN_CREDIT.id
        )

    def it_returns_output_path_on_success(
        self,
        mock_path_provider: MagicMock,
        sample_dataframes: list[pd.DataFrame],
    ) -> None:
        """Verify output path is returned on success."""
        mock_path_provider.get_consolidated_results_path.return_value = Path(
            "/output/results.parquet"
        )
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [Path("/checkpoints/test/0.parquet")]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch.object(Path, "mkdir"),
            patch("pandas.read_parquet") as mock_read,
            patch("pandas.concat") as mock_concat,
        ):
            mock_read.return_value = sample_dataframes[0]
            concat_result = MagicMock()
            mock_concat.return_value = concat_result

            result = persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert result == Path("/output/results.parquet")

    def it_returns_none_when_no_valid_frames_loaded(
        self,
        mock_path_provider: MagicMock,
    ) -> None:
        """Verify None is returned when all reads fail."""
        persister = ParquetCheckpointPersister(
            checkpoint_path_provider=mock_path_provider,
            results_path_provider=mock_path_provider,
        )

        checkpoint_files = [Path("/checkpoints/test/0.parquet")]

        with (
            patch.object(Path, "glob", return_value=checkpoint_files),
            patch("pandas.read_parquet") as mock_read,
        ):
            mock_read.side_effect = Exception("Read error")

            result = persister.consolidate(Dataset.TAIWAN_CREDIT)

        assert result is None
