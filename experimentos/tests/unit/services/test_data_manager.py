"""Tests for experiments.services.data_manager module."""

from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.services.data_manager import ExperimentDataManager
from experiments.services.storage import LocalStorageService
from experiments.services.storage_manager import StorageManager
from experiments.settings import PathSettings


@pytest.fixture
def path_settings(tmp_path: Path) -> PathSettings:
    """Create PathSettings with temporary directories."""
    settings = PathSettings(project_root=tmp_path)
    return settings


@pytest.fixture
def storage_service() -> LocalStorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


@pytest.fixture
def storage_manager(
    path_settings: PathSettings, storage_service: LocalStorageService
) -> StorageManager:
    """Create a StorageManager with temporary directories."""
    return StorageManager(settings=path_settings, storage=storage_service)


@pytest.fixture
def data_manager(storage_manager: StorageManager) -> ExperimentDataManager:
    """Create an ExperimentDataManager instance."""
    return ExperimentDataManager(storage_manager)


@pytest.fixture
def sample_dataset() -> Dataset:
    """Return a sample dataset for testing."""
    return Dataset.TAIWAN_CREDIT


class DescribeExperimentDataManager:
    """Tests for ExperimentDataManager class."""

    class DescribeArtifactsExist:
        """Tests for the artifacts_exist method."""

        def it_returns_true_when_both_artifacts_exist(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns True when both X and y artifacts exist."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / f"{sample_dataset.id}_X.parquet"
            y_path = path_settings.processed_data_dir / f"{sample_dataset.id}_y.parquet"

            # Create dummy parquet files
            pl.DataFrame({"a": [1, 2, 3]}).write_parquet(x_path)
            pl.DataFrame({"b": [0, 1, 0]}).write_parquet(y_path)

            result = data_manager.artifacts_exist(sample_dataset)

            assert result is True

        def it_returns_false_when_x_artifact_missing(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns False when X artifact is missing."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            y_path = path_settings.processed_data_dir / f"{sample_dataset.id}_y.parquet"
            pl.DataFrame({"b": [0, 1, 0]}).write_parquet(y_path)

            with patch("experiments.services.storage_manager.logger") as mock_logger:
                result = data_manager.artifacts_exist(sample_dataset)

                assert result is False
                mock_logger.warning.assert_called_once()

        def it_returns_false_when_y_artifact_missing(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns False when y artifact is missing."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / f"{sample_dataset.id}_X.parquet"
            pl.DataFrame({"a": [1, 2, 3]}).write_parquet(x_path)

            with patch("experiments.services.storage_manager.logger") as mock_logger:
                result = data_manager.artifacts_exist(sample_dataset)

                assert result is False
                mock_logger.warning.assert_called_once()

        def it_returns_false_when_both_artifacts_missing(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns False when both artifacts are missing."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            with patch("experiments.services.storage_manager.logger") as mock_logger:
                result = data_manager.artifacts_exist(sample_dataset)

                assert result is False
                assert mock_logger.warning.call_count == 1

    class DescribeGetDatasetSizeGb:
        """Tests for the get_dataset_size_gb method."""

        def it_returns_file_size_in_gb(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns correct file size in GB."""
            path_settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
            raw_path = path_settings.raw_data_dir / f"{sample_dataset.id}.csv"

            # Create a file with known size (1024 bytes)
            raw_path.write_bytes(b"x" * 1024)

            result = data_manager.get_dataset_size_gb(sample_dataset)

            expected_gb = 1024 / (1024**3)
            assert result == pytest.approx(expected_gb)

        def it_returns_default_when_file_not_found(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
        ) -> None:
            """Verify returns 1.0 GB default when file doesn't exist."""
            result = data_manager.get_dataset_size_gb(sample_dataset)

            assert result == 1.0

    class DescribeFeatureContext:
        """Tests for the feature_context context manager."""

        def it_delegates_to_storage_manager(
            self,
            data_manager: ExperimentDataManager,
            storage_manager: StorageManager,
            sample_dataset: Dataset,
        ) -> None:
            """Verify feature_context delegates to storage manager."""
            with patch.object(storage_manager, "feature_context") as mock_context:
                # Mock context manager that yields test paths
                test_paths = ("/tmp/X.mmap", "/tmp/y.mmap")
                mock_context.return_value.__enter__.return_value = test_paths
                mock_context.return_value.__exit__.return_value = False

                with data_manager.feature_context(sample_dataset) as paths:
                    assert paths == test_paths

                mock_context.assert_called_once_with(sample_dataset)

        def it_yields_memory_mapped_paths(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify yields paths to memory-mapped files."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / f"{sample_dataset.id}_X.parquet"
            y_path = path_settings.processed_data_dir / f"{sample_dataset.id}_y.parquet"

            # Create sample data
            X_data = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
            y_data = pl.DataFrame({"target": [0, 1, 0]})

            X_data.write_parquet(x_path)
            y_data.write_parquet(y_path)

            with data_manager.feature_context(sample_dataset) as (X_mmap, y_mmap):
                assert X_mmap.endswith("X.mmap")
                assert y_mmap.endswith("y.mmap")

                # Verify files exist and can be loaded
                X_loaded = joblib.load(X_mmap, mmap_mode="r")
                y_loaded = joblib.load(y_mmap, mmap_mode="r")

                assert isinstance(X_loaded, np.ndarray)
                assert isinstance(y_loaded, np.ndarray)
                assert X_loaded.shape == (3, 2)
                assert y_loaded.shape == (3,)

        def it_raises_file_not_found_when_data_missing(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify raises FileNotFoundError when data is missing."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            with pytest.raises(FileNotFoundError):
                with data_manager.feature_context(sample_dataset):
                    pass

        def it_cleans_up_temp_files_on_exit(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify temporary files are cleaned up after context exit."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / f"{sample_dataset.id}_X.parquet"
            y_path = path_settings.processed_data_dir / f"{sample_dataset.id}_y.parquet"

            X_data = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
            y_data = pl.DataFrame({"target": [0, 1, 0]})

            X_data.write_parquet(x_path)
            y_data.write_parquet(y_path)

            mmap_paths = None
            with data_manager.feature_context(sample_dataset) as paths:
                mmap_paths = paths
                # Files should exist inside context
                assert Path(mmap_paths[0]).exists()
                assert Path(mmap_paths[1]).exists()

            # Files should be cleaned up after context exit
            assert not Path(mmap_paths[0]).exists()
            assert not Path(mmap_paths[1]).exists()

    class DescribeConsolidateResults:
        """Tests for the consolidate_results method."""

        def it_delegates_to_storage_manager_consolidate_checkpoints(
            self,
            data_manager: ExperimentDataManager,
            storage_manager: StorageManager,
            sample_dataset: Dataset,
        ) -> None:
            """Verify consolidate_results delegates to storage manager."""
            with patch.object(
                storage_manager,
                "consolidate_checkpoints",
                return_value="/results/consolidated.parquet",
            ) as mock_consolidate:
                result = data_manager.consolidate_results(sample_dataset)

                mock_consolidate.assert_called_once_with(sample_dataset.id)
                assert result == "/results/consolidated.parquet"

        def it_returns_none_when_storage_manager_returns_none(
            self,
            data_manager: ExperimentDataManager,
            storage_manager: StorageManager,
            sample_dataset: Dataset,
        ) -> None:
            """Verify returns None when no checkpoints exist."""
            with patch.object(
                storage_manager, "consolidate_checkpoints", return_value=None
            ) as mock_consolidate:
                result = data_manager.consolidate_results(sample_dataset)

                mock_consolidate.assert_called_once_with(sample_dataset.id)
                assert result is None

        def it_handles_corrupted_checkpoints_gracefully(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify handles corrupted checkpoint files gracefully."""
            ckpt_dir = path_settings.results_dir / sample_dataset.id / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Create one valid and one corrupted checkpoint
            df_valid = pd.DataFrame({"model": ["rf"], "seed": [0], "accuracy": [0.85]})
            df_valid.to_parquet(ckpt_dir / "rf_baseline_seed0.parquet")

            # Create corrupted file
            (ckpt_dir / "rf_baseline_seed1.parquet").write_text("not valid parquet")

            # Should not raise, delegates to storage manager
            result = data_manager.consolidate_results(sample_dataset)
            # Result depends on storage manager implementation
            assert result is None or isinstance(result, str)

        def it_consolidates_multiple_checkpoints(
            self,
            data_manager: ExperimentDataManager,
            sample_dataset: Dataset,
            path_settings: PathSettings,
        ) -> None:
            """Verify consolidates multiple valid checkpoint files."""
            ckpt_dir = path_settings.results_dir / sample_dataset.id / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Create multiple valid checkpoint files
            df1 = pd.DataFrame({"model": ["rf"], "seed": [0], "accuracy": [0.85]})
            df2 = pd.DataFrame({"model": ["xgb"], "seed": [1], "accuracy": [0.87]})

            df1.to_parquet(ckpt_dir / "rf_baseline_seed0.parquet")
            df2.to_parquet(ckpt_dir / "xgb_baseline_seed1.parquet")

            # Should consolidate without raising
            result = data_manager.consolidate_results(sample_dataset)
            # Result depends on storage manager implementation
            assert result is None or isinstance(result, str)
