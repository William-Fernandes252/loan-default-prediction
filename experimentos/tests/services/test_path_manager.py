"""Tests for experiments.services.path_manager module."""

from datetime import datetime
from pathlib import Path
import time

import pytest

from experiments.services.path_manager import PathManager
from experiments.settings import PathSettings


@pytest.fixture
def path_settings(tmp_path: Path) -> PathSettings:
    """Create PathSettings with temporary directories."""
    return PathSettings(project_root=tmp_path)


@pytest.fixture
def path_manager(path_settings: PathSettings) -> PathManager:
    """Create a PathManager with temporary directories."""
    return PathManager(path_settings)


class DescribePathManager:
    """Tests for PathManager class."""

    class DescribeRawDataPath:
        """Tests for get_raw_data_path method."""

        def it_returns_correct_path_for_dataset(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns correct raw data path."""
            result = path_manager.get_raw_data_path("taiwan_credit")

            expected = path_settings.raw_data_dir / "taiwan_credit.csv"
            assert result == expected

        def it_handles_different_dataset_ids(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify handles different dataset identifiers."""
            result = path_manager.get_raw_data_path("lending_club")

            expected = path_settings.raw_data_dir / "lending_club.csv"
            assert result == expected

    class DescribeInterimDataPath:
        """Tests for get_interim_data_path method."""

        def it_returns_correct_parquet_path(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns correct interim parquet path."""
            result = path_manager.get_interim_data_path("taiwan_credit")

            expected = path_settings.interim_data_dir / "taiwan_credit.parquet"
            assert result == expected

    class DescribeFeaturePaths:
        """Tests for get_feature_paths method."""

        def it_returns_x_and_y_paths(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns dictionary with X and y paths."""
            result = path_manager.get_feature_paths("taiwan_credit")

            assert "X" in result
            assert "y" in result
            assert result["X"] == path_settings.processed_data_dir / "taiwan_credit_X.parquet"
            assert result["y"] == path_settings.processed_data_dir / "taiwan_credit_y.parquet"

    class DescribeCheckpointPath:
        """Tests for get_checkpoint_path method."""

        def it_returns_correct_checkpoint_path(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns correct checkpoint file path."""
            result = path_manager.get_checkpoint_path(
                dataset_id="taiwan_credit",
                model_id="random_forest",
                technique_id="smote",
                seed=42,
            )

            expected = (
                path_settings.results_dir
                / "taiwan_credit"
                / "checkpoints"
                / "random_forest_smote_seed42.parquet"
            )
            assert result == expected

        def it_creates_checkpoint_directory(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify creates checkpoint directory if needed."""
            result = path_manager.get_checkpoint_path(
                dataset_id="taiwan_credit",
                model_id="rf",
                technique_id="baseline",
                seed=0,
            )

            assert result.parent.exists()
            assert result.parent.name == "checkpoints"

    class DescribeConsolidatedResultsPath:
        """Tests for consolidated results path methods."""

        def it_returns_timestamped_path(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns path with timestamp."""
            before = datetime.now()
            result = path_manager.get_consolidated_results_path("taiwan_credit")
            after = datetime.now()

            # Check path structure
            assert result.parent == path_settings.results_dir / "taiwan_credit"
            assert result.suffix == ".parquet"

            # Check timestamp format (YYYYMMDD_HHMMSS)
            filename = result.stem
            assert len(filename) == 15  # 8 + 1 + 6
            assert filename[8] == "_"

        def it_creates_results_directory(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify creates results directory if needed."""
            result = path_manager.get_consolidated_results_path("taiwan_credit")

            assert result.parent.exists()

        def it_returns_none_when_no_results_exist(
            self,
            path_manager: PathManager,
        ) -> None:
            """Verify returns None when no consolidated results exist."""
            result = path_manager.get_latest_consolidated_results_path("nonexistent")

            assert result is None

        def it_returns_latest_file_by_modification_time(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns most recently modified file."""
            # Create dataset results directory
            dataset_dir = path_settings.results_dir / "taiwan_credit"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Create files with different timestamps
            older_file = dataset_dir / "20230101_120000.parquet"
            newer_file = dataset_dir / "20230102_120000.parquet"

            older_file.write_text("older")
            time.sleep(0.01)  # Ensure different mtime
            newer_file.write_text("newer")

            result = path_manager.get_latest_consolidated_results_path("taiwan_credit")

            assert result == newer_file

        def it_returns_none_when_directory_is_empty(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns None when directory exists but has no matching files."""
            dataset_dir = path_settings.results_dir / "taiwan_credit"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            result = path_manager.get_latest_consolidated_results_path("taiwan_credit")

            assert result is None

    class DescribeDatasetResultsDir:
        """Tests for get_dataset_results_dir method."""

        def it_returns_correct_directory_path(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns correct results directory path."""
            result = path_manager.get_dataset_results_dir("taiwan_credit")

            expected = path_settings.results_dir / "taiwan_credit"
            assert result == expected

        def it_creates_directory_when_requested(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify creates directory when create=True."""
            result = path_manager.get_dataset_results_dir("taiwan_credit", create=True)

            assert result.exists()

        def it_does_not_create_directory_by_default(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify does not create directory by default."""
            result = path_manager.get_dataset_results_dir("new_dataset")

            assert not result.exists()

    class DescribeDatasetFiguresDir:
        """Tests for get_dataset_figures_dir method."""

        def it_returns_correct_figures_path(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns correct figures directory path."""
            result = path_manager.get_dataset_figures_dir("taiwan_credit")

            expected = path_settings.figures_dir / "taiwan_credit"
            assert result == expected

        def it_creates_directory_when_requested(
            self,
            path_manager: PathManager,
        ) -> None:
            """Verify creates directory when create=True."""
            result = path_manager.get_dataset_figures_dir("taiwan_credit", create=True)

            assert result.exists()

    class DescribeModelsDir:
        """Tests for models_dir property."""

        def it_returns_models_directory(
            self,
            path_manager: PathManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify returns models directory path."""
            result = path_manager.models_dir

            assert result == path_settings.models_dir
