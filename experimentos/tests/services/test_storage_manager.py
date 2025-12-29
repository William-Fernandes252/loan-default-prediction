"""Tests for experiments.services.storage_manager module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.services.storage import LocalStorageService
from experiments.services.storage_manager import StorageManager
from experiments.settings import PathSettings


@pytest.fixture
def path_settings(tmp_path: Path) -> PathSettings:
    """Create PathSettings with temporary directories."""
    return PathSettings(project_root=tmp_path)


@pytest.fixture
def local_storage() -> LocalStorageService:
    """Create a LocalStorageService instance."""
    return LocalStorageService()


@pytest.fixture
def storage_manager(
    path_settings: PathSettings, local_storage: LocalStorageService
) -> StorageManager:
    """Create a StorageManager with local storage."""
    return StorageManager(settings=path_settings, storage=local_storage)


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage service."""
    return MagicMock()


@pytest.fixture
def storage_manager_with_mock(
    path_settings: PathSettings, mock_storage: MagicMock
) -> StorageManager:
    """Create a StorageManager with mocked storage."""
    return StorageManager(settings=path_settings, storage=mock_storage)


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_dataset() -> Dataset:
    """Return a sample dataset for testing."""
    return Dataset.TAIWAN_CREDIT


class DescribeStorageManager:
    """Tests for the StorageManager class."""

    class DescribeInit:
        """Tests for initialization."""

        def it_initializes_with_settings_and_storage(
            self, path_settings: PathSettings, local_storage: LocalStorageService
        ) -> None:
            """Verify initializes with settings and storage service."""
            manager = StorageManager(settings=path_settings, storage=local_storage)

            assert manager._settings is path_settings
            assert manager._storage is local_storage

        def it_exposes_storage_property(
            self, storage_manager: StorageManager, local_storage: LocalStorageService
        ) -> None:
            """Verify storage property returns the storage service."""
            assert storage_manager.storage is local_storage

    class DescribeRawDataUri:
        """Tests for get_raw_data_uri method."""

        def it_returns_uri_for_raw_csv(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify returns URI to raw CSV file."""
            uri = storage_manager.get_raw_data_uri("taiwan_credit")

            expected_path = path_settings.raw_data_dir / "taiwan_credit.csv"
            assert f"file://{expected_path}" == uri

    class DescribeInterimDataUri:
        """Tests for get_interim_data_uri method."""

        def it_returns_uri_for_interim_parquet(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify returns URI to interim parquet file."""
            uri = storage_manager.get_interim_data_uri("taiwan_credit")

            expected_path = path_settings.interim_data_dir / "taiwan_credit.parquet"
            assert f"file://{expected_path}" == uri

    class DescribeFeatureUris:
        """Tests for get_feature_uris method."""

        def it_returns_uris_for_x_and_y(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify returns URIs for X and y feature files."""
            uris = storage_manager.get_feature_uris("taiwan_credit")

            assert "X" in uris
            assert "y" in uris
            assert "taiwan_credit_X.parquet" in uris["X"]
            assert "taiwan_credit_y.parquet" in uris["y"]

    class DescribeCheckpointUri:
        """Tests for get_checkpoint_uri method."""

        def it_returns_checkpoint_uri(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify returns URI for specific experiment checkpoint."""
            uri = storage_manager.get_checkpoint_uri(
                dataset_id="taiwan_credit",
                model_id="random_forest",
                technique_id="baseline",
                seed=42,
            )

            assert "taiwan_credit" in uri
            assert "checkpoints" in uri
            assert "random_forest_baseline_seed42.parquet" in uri

        def it_creates_checkpoint_directory(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
        ) -> None:
            """Verify creates checkpoint directory."""
            storage_manager.get_checkpoint_uri(
                dataset_id="taiwan_credit",
                model_id="rf",
                technique_id="smote",
                seed=1,
            )

            ckpt_dir = path_settings.results_dir / "taiwan_credit" / "checkpoints"
            assert ckpt_dir.exists()

    class DescribeCheckpointsDirUri:
        """Tests for get_checkpoints_dir_uri method."""

        def it_returns_checkpoints_directory_uri(self, storage_manager: StorageManager) -> None:
            """Verify returns URI to checkpoints directory."""
            uri = storage_manager.get_checkpoints_dir_uri("taiwan_credit")

            assert "taiwan_credit" in uri
            assert "checkpoints" in uri

    class DescribeConsolidatedResultsUri:
        """Tests for get_consolidated_results_uri method."""

        def it_returns_timestamped_uri(self, storage_manager: StorageManager) -> None:
            """Verify returns URI with timestamp."""
            uri = storage_manager.get_consolidated_results_uri("taiwan_credit")

            assert "taiwan_credit" in uri
            assert ".parquet" in uri
            # Should contain timestamp pattern YYYYMMDD_HHMMSS
            import re

            assert re.search(r"\d{8}_\d{6}\.parquet", uri)

        def it_creates_results_directory(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify creates results directory."""
            storage_manager.get_consolidated_results_uri("taiwan_credit")

            results_dir = path_settings.results_dir / "taiwan_credit"
            assert results_dir.exists()

    class DescribeLatestConsolidatedResultsUri:
        """Tests for get_latest_consolidated_results_uri method."""

        def it_returns_none_when_no_results(self, storage_manager: StorageManager) -> None:
            """Verify returns None when no results exist."""
            uri = storage_manager.get_latest_consolidated_results_uri("taiwan_credit")

            assert uri is None

        def it_returns_latest_file(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify returns most recent results file."""
            results_dir = path_settings.results_dir / "taiwan_credit"
            results_dir.mkdir(parents=True)

            # Create files with different timestamps
            (results_dir / "20251225_100000.parquet").touch()
            (results_dir / "20251225_120000.parquet").touch()
            (results_dir / "20251225_110000.parquet").touch()

            uri = storage_manager.get_latest_consolidated_results_uri("taiwan_credit")

            assert "20251225_120000.parquet" in uri  # Latest by alphabetical order

    class DescribeDatasetResultsDirUri:
        """Tests for get_dataset_results_dir_uri method."""

        def it_returns_results_directory_uri(self, storage_manager: StorageManager) -> None:
            """Verify returns URI to results directory."""
            uri = storage_manager.get_dataset_results_dir_uri("taiwan_credit")

            assert "taiwan_credit" in uri

        def it_creates_directory_when_requested(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify creates directory when create=True."""
            storage_manager.get_dataset_results_dir_uri("taiwan_credit", create=True)

            results_dir = path_settings.results_dir / "taiwan_credit"
            assert results_dir.exists()

    class DescribeDatasetFiguresDirUri:
        """Tests for get_dataset_figures_dir_uri method."""

        def it_returns_figures_directory_uri(self, storage_manager: StorageManager) -> None:
            """Verify returns URI to figures directory."""
            uri = storage_manager.get_dataset_figures_dir_uri("taiwan_credit")

            assert "taiwan_credit" in uri

    class DescribeModelsDirUri:
        """Tests for get_models_dir_uri method."""

        def it_returns_models_directory_uri(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify returns URI to models directory."""
            uri = storage_manager.get_models_dir_uri()

            expected_path = path_settings.models_dir
            assert f"file://{expected_path}" == uri


class DescribeStorageManagerDataOperations:
    """Tests for StorageManager data operations."""

    class DescribeReadRawData:
        """Tests for read_raw_data method."""

        def it_reads_csv_from_raw_data_path(
            self, storage_manager: StorageManager, path_settings: PathSettings
        ) -> None:
            """Verify reads CSV from raw data directory."""
            path_settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = path_settings.raw_data_dir / "taiwan_credit.csv"
            csv_path.write_text("a,b,c\n1,2,3")

            df = storage_manager.read_raw_data("taiwan_credit")

            assert df.shape == (1, 3)
            assert df.columns == ["a", "b", "c"]

        def it_passes_kwargs_to_read_csv(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify passes kwargs to storage.read_csv."""
            mock_storage.read_csv.return_value = pl.DataFrame()

            storage_manager_with_mock.read_raw_data("taiwan_credit", separator=";", skip_rows=1)

            mock_storage.read_csv.assert_called_once()
            call_kwargs = mock_storage.read_csv.call_args[1]
            assert call_kwargs["separator"] == ";"
            assert call_kwargs["skip_rows"] == 1

    class DescribeWriteInterimData:
        """Tests for write_interim_data method."""

        def it_writes_parquet_to_interim_path(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes DataFrame to interim directory."""
            path_settings.interim_data_dir.mkdir(parents=True, exist_ok=True)

            uri = storage_manager.write_interim_data(sample_dataframe, "taiwan_credit")

            assert "taiwan_credit.parquet" in uri
            parquet_path = path_settings.interim_data_dir / "taiwan_credit.parquet"
            assert parquet_path.exists()

    class DescribeReadFeatures:
        """Tests for read_features method."""

        def it_reads_x_and_y_dataframes(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify reads both X and y feature files."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / "taiwan_credit_X.parquet"
            y_path = path_settings.processed_data_dir / "taiwan_credit_y.parquet"

            sample_dataframe.write_parquet(x_path)
            pl.DataFrame({"target": [0, 1, 0]}).write_parquet(y_path)

            X, y = storage_manager.read_features("taiwan_credit")

            assert X.shape == sample_dataframe.shape
            assert y.shape == (3, 1)

    class DescribeWriteFeatures:
        """Tests for write_features method."""

        def it_writes_x_and_y_dataframes(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes both X and y feature files."""
            y = pl.DataFrame({"target": [0, 1, 0]})

            uris = storage_manager.write_features(sample_dataframe, y, "taiwan_credit")

            assert "X" in uris
            assert "y" in uris

            x_path = path_settings.processed_data_dir / "taiwan_credit_X.parquet"
            y_path = path_settings.processed_data_dir / "taiwan_credit_y.parquet"
            assert x_path.exists()
            assert y_path.exists()

    class DescribeFeaturesExist:
        """Tests for features_exist method."""

        def it_returns_true_when_both_exist(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify returns True when both feature files exist."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / "taiwan_credit_X.parquet"
            y_path = path_settings.processed_data_dir / "taiwan_credit_y.parquet"
            sample_dataframe.write_parquet(x_path)
            sample_dataframe.write_parquet(y_path)

            assert storage_manager.features_exist("taiwan_credit") is True

        def it_returns_false_when_x_missing(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify returns False when X file is missing."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            y_path = path_settings.processed_data_dir / "taiwan_credit_y.parquet"
            sample_dataframe.write_parquet(y_path)

            assert storage_manager.features_exist("taiwan_credit") is False

        def it_returns_false_when_y_missing(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify returns False when y file is missing."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / "taiwan_credit_X.parquet"
            sample_dataframe.write_parquet(x_path)

            assert storage_manager.features_exist("taiwan_credit") is False

    class DescribeArtifactsExist:
        """Tests for artifacts_exist method (DataProvider protocol)."""

        def it_returns_true_when_artifacts_exist(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
            sample_dataset: Dataset,
        ) -> None:
            """Verify returns True for existing artifacts."""
            path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_path = path_settings.processed_data_dir / f"{sample_dataset.id}_X.parquet"
            y_path = path_settings.processed_data_dir / f"{sample_dataset.id}_y.parquet"
            sample_dataframe.write_parquet(x_path)
            sample_dataframe.write_parquet(y_path)

            assert storage_manager.artifacts_exist(sample_dataset) is True

        def it_logs_warning_when_missing(
            self,
            storage_manager: StorageManager,
            sample_dataset: Dataset,
        ) -> None:
            """Verify logs warning when artifacts are missing."""
            with patch("experiments.services.storage_manager.logger") as mock_logger:
                result = storage_manager.artifacts_exist(sample_dataset)

                assert result is False
                mock_logger.warning.assert_called_once()


class DescribeStorageManagerCheckpointOperations:
    """Tests for StorageManager checkpoint operations."""

    class DescribeWriteCheckpoint:
        """Tests for write_checkpoint method."""

        def it_writes_checkpoint_to_uri(
            self,
            storage_manager_with_mock: StorageManager,
            mock_storage: MagicMock,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify writes checkpoint DataFrame to URI."""
            uri = "file:///tmp/checkpoint.parquet"

            storage_manager_with_mock.write_checkpoint(sample_dataframe, uri)

            mock_storage.write_parquet.assert_called_once_with(sample_dataframe, uri)

    class DescribeReadCheckpoint:
        """Tests for read_checkpoint method."""

        def it_reads_checkpoint_from_uri(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify reads checkpoint DataFrame from URI."""
            mock_storage.read_parquet.return_value = pl.DataFrame()
            uri = "file:///tmp/checkpoint.parquet"

            storage_manager_with_mock.read_checkpoint(uri)

            mock_storage.read_parquet.assert_called_once_with(uri)

    class DescribeCheckpointExists:
        """Tests for checkpoint_exists method."""

        def it_returns_storage_exists_result(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify returns result from storage.exists."""
            mock_storage.exists.return_value = True
            uri = "file:///tmp/checkpoint.parquet"

            result = storage_manager_with_mock.checkpoint_exists(uri)

            assert result is True
            mock_storage.exists.assert_called_once_with(uri)

    class DescribeDeleteCheckpoint:
        """Tests for delete_checkpoint method."""

        def it_deletes_checkpoint_via_storage(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify deletes checkpoint via storage service."""
            uri = "file:///tmp/checkpoint.parquet"

            storage_manager_with_mock.delete_checkpoint(uri)

            mock_storage.delete.assert_called_once_with(uri)

    class DescribeListCheckpoints:
        """Tests for list_checkpoints method."""

        def it_lists_parquet_files_in_checkpoint_dir(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify lists parquet files in checkpoint directory."""
            mock_storage.list_files.return_value = [
                "file:///path/ckpt1.parquet",
                "file:///path/ckpt2.parquet",
            ]
            mock_storage.makedirs = MagicMock()

            checkpoints = storage_manager_with_mock.list_checkpoints("taiwan_credit")

            assert len(checkpoints) == 2
            mock_storage.list_files.assert_called_once()

    class DescribeConsolidateCheckpoints:
        """Tests for consolidate_checkpoints method."""

        def it_returns_none_when_no_checkpoints(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify returns None when no checkpoints exist."""
            mock_storage.list_files.return_value = []

            result = storage_manager_with_mock.consolidate_checkpoints("taiwan_credit")

            assert result is None

        def it_concatenates_and_writes_consolidated_file(
            self,
            storage_manager: StorageManager,
            path_settings: PathSettings,
            sample_dataframe: pl.DataFrame,
        ) -> None:
            """Verify concatenates checkpoints and writes result."""
            # Create checkpoint files
            ckpt_dir = path_settings.results_dir / "taiwan_credit" / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            df1 = sample_dataframe.with_columns(pl.lit(1).alias("seed"))
            df2 = sample_dataframe.with_columns(pl.lit(2).alias("seed"))

            df1.write_parquet(ckpt_dir / "ckpt1.parquet")
            df2.write_parquet(ckpt_dir / "ckpt2.parquet")

            result_uri = storage_manager.consolidate_checkpoints("taiwan_credit")

            assert result_uri is not None
            assert ".parquet" in result_uri

            # Verify consolidated file contains both checkpoints
            _, path = storage_manager.storage.parse_uri(result_uri)
            consolidated = pl.read_parquet(path)
            assert len(consolidated) == 6  # 3 rows * 2 checkpoints


class DescribeStorageManagerFeatureContext:
    """Tests for StorageManager feature_context method."""

    def it_creates_memory_mapped_files(
        self,
        storage_manager: StorageManager,
        path_settings: PathSettings,
        sample_dataframe: pl.DataFrame,
        sample_dataset: Dataset,
    ) -> None:
        """Verify creates memory-mapped files for parallel access."""
        path_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

        x_path = path_settings.processed_data_dir / f"{sample_dataset.id}_X.parquet"
        y_path = path_settings.processed_data_dir / f"{sample_dataset.id}_y.parquet"
        sample_dataframe.write_parquet(x_path)
        pl.DataFrame({"target": [0, 1, 0]}).write_parquet(y_path)

        with storage_manager.feature_context(sample_dataset) as (x_mmap, y_mmap):
            assert Path(x_mmap).exists()
            assert Path(y_mmap).exists()
            assert x_mmap.endswith(".mmap")
            assert y_mmap.endswith(".mmap")

    def it_raises_when_data_missing(
        self, storage_manager: StorageManager, sample_dataset: Dataset
    ) -> None:
        """Verify raises FileNotFoundError when data is missing."""
        with pytest.raises(FileNotFoundError):
            with storage_manager.feature_context(sample_dataset):
                pass


class DescribeStorageManagerModelOperations:
    """Tests for StorageManager model operations."""

    class DescribeSaveModel:
        """Tests for save_model method."""

        def it_saves_model_via_storage(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify saves model using storage.write_joblib."""
            model = {"type": "dummy_model"}
            uri = "file:///models/model.joblib"

            storage_manager_with_mock.save_model(model, uri)

            mock_storage.write_joblib.assert_called_once_with(model, uri)

    class DescribeLoadModel:
        """Tests for load_model method."""

        def it_loads_model_via_storage(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify loads model using storage.read_joblib."""
            mock_storage.read_joblib.return_value = {"type": "dummy_model"}
            uri = "file:///models/model.joblib"

            model = storage_manager_with_mock.load_model(uri)

            assert model == {"type": "dummy_model"}
            mock_storage.read_joblib.assert_called_once_with(uri)

    class DescribeSaveModelMetadata:
        """Tests for save_model_metadata method."""

        def it_saves_metadata_as_json(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify saves metadata using storage.write_json."""
            metadata = {"accuracy": 0.95, "model_type": "rf"}
            uri = "file:///models/metadata.json"

            storage_manager_with_mock.save_model_metadata(metadata, uri)

            mock_storage.write_json.assert_called_once_with(metadata, uri)

    class DescribeLoadModelMetadata:
        """Tests for load_model_metadata method."""

        def it_loads_metadata_from_json(
            self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
        ) -> None:
            """Verify loads metadata using storage.read_json."""
            mock_storage.read_json.return_value = {"accuracy": 0.95}
            uri = "file:///models/metadata.json"

            metadata = storage_manager_with_mock.load_model_metadata(uri)

            assert metadata == {"accuracy": 0.95}
            mock_storage.read_json.assert_called_once_with(uri)


class DescribeStorageManagerGetDatasetSizeGb:
    """Tests for get_dataset_size_gb method."""

    def it_returns_size_in_gb(
        self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
    ) -> None:
        """Verify returns file size in gigabytes."""
        mock_storage.get_size_bytes.return_value = 1024**3  # 1 GB
        dataset = Dataset.TAIWAN_CREDIT

        size = storage_manager_with_mock.get_dataset_size_gb(dataset)

        assert size == 1.0

    def it_returns_default_on_error(
        self, storage_manager_with_mock: StorageManager, mock_storage: MagicMock
    ) -> None:
        """Verify returns 1.0 GB default when file doesn't exist."""
        mock_storage.get_size_bytes.side_effect = Exception("File not found")
        dataset = Dataset.TAIWAN_CREDIT

        size = storage_manager_with_mock.get_dataset_size_gb(dataset)

        assert size == 1.0
