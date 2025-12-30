"""Tests for experiments.core.experiment.persisters module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.core.data import Dataset
from experiments.core.experiment.persisters import (
    CompositeExperimentPersister,
    ParquetExperimentPersister,
)
from experiments.core.experiment.protocols import (
    DataPaths,
    ExperimentContext,
    ExperimentIdentity,
    TrainingConfig,
)
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.storage import StorageService
from experiments.services.storage.local import LocalStorageService


@pytest.fixture
def storage() -> StorageService:
    """Create a local storage service for testing."""
    return LocalStorageService()


@pytest.fixture
def sample_context() -> ExperimentContext:
    """Create a sample experiment context."""
    identity = ExperimentIdentity(
        dataset=Dataset.TAIWAN_CREDIT,
        model_type=ModelType.RANDOM_FOREST,
        technique=Technique.BASELINE,
        seed=42,
    )
    data_paths = DataPaths(
        X_path="/data/X.mmap",
        y_path="/data/y.mmap",
    )
    training_config = TrainingConfig(
        cv_folds=5,
        cost_grids=[],
    )
    return ExperimentContext(
        identity=identity,
        data=data_paths,
        config=training_config,
        checkpoint_uri="/tmp/test_checkpoint.parquet",
    )


@pytest.fixture
def sample_metrics() -> dict[str, Any]:
    """Create sample metrics dictionary."""
    return {
        "accuracy_balanced": 0.85,
        "f1_score": 0.80,
        "roc_auc": 0.90,
        "dataset": "taiwan_credit",
        "seed": 42,
        "model": "random_forest",
        "technique": "baseline",
    }


class DescribeParquetExperimentPersister:
    """Tests for ParquetExperimentPersister class."""

    def it_initializes_without_versioning_service(self, storage: StorageService) -> None:
        """Verify persister can be initialized without versioning service."""
        persister = ParquetExperimentPersister(storage=storage)

        assert persister._model_versioning_service_factory is None

    def it_initializes_with_versioning_service_factory(self, storage: StorageService) -> None:
        """Verify persister stores versioning service factory."""
        mock_factory = MagicMock()
        persister = ParquetExperimentPersister(
            storage=storage, model_versioning_service_factory=mock_factory
        )

        assert persister._model_versioning_service_factory is mock_factory


class DescribeParquetExperimentPersisterSaveCheckpoint:
    """Tests for ParquetExperimentPersister.save_checkpoint() method."""

    def it_saves_metrics_to_parquet(
        self,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
        storage: StorageService,
    ) -> None:
        """Verify metrics are saved to parquet file."""
        checkpoint_uri = str(tmp_path / "checkpoint.parquet")
        persister = ParquetExperimentPersister(storage=storage)

        persister.save_checkpoint(sample_metrics, checkpoint_uri)

        assert Path(checkpoint_uri).exists()

        # Read back and verify
        df = pd.read_parquet(checkpoint_uri)
        assert len(df) == 1
        assert df.iloc[0]["accuracy_balanced"] == 0.85
        assert df.iloc[0]["dataset"] == "taiwan_credit"

    def it_creates_parent_directories_implicitly(
        self,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
        storage: StorageService,
    ) -> None:
        """Verify parent directories are created if needed."""
        checkpoint_uri = str(tmp_path / "subdir" / "nested" / "checkpoint.parquet")

        persister = ParquetExperimentPersister(storage=storage)
        persister.save_checkpoint(sample_metrics, checkpoint_uri)

        assert Path(checkpoint_uri).exists()


class DescribeParquetExperimentPersisterSaveModel:
    """Tests for ParquetExperimentPersister.save_model() method."""

    def it_does_nothing_when_no_versioning_service(
        self,
        sample_context: ExperimentContext,
        storage: StorageService,
    ) -> None:
        """Verify no error when versioning service is None."""
        persister = ParquetExperimentPersister(storage=storage)
        model = LogisticRegression()

        # Should not raise
        persister.save_model(model, sample_context)

    def it_calls_versioning_factory_to_create_service(
        self,
        sample_context: ExperimentContext,
        storage: StorageService,
    ) -> None:
        """Verify factory is used to create versioning service and save model."""
        mock_service = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_model_versioning_service.return_value = mock_service

        persister = ParquetExperimentPersister(
            storage=storage, model_versioning_service_factory=mock_factory
        )
        model = LogisticRegression()

        persister.save_model(model, sample_context)

        # Verify factory was called with correct parameters
        mock_factory.get_model_versioning_service.assert_called_once_with(
            dataset_id=sample_context.identity.dataset.id,
            model_type=sample_context.identity.model_type,
            technique=sample_context.identity.technique,
        )
        # Verify service save_model was called
        mock_service.save_model.assert_called_once_with(model, None)

    def it_handles_versioning_factory_errors_gracefully(
        self,
        sample_context: ExperimentContext,
        storage: StorageService,
    ) -> None:
        """Verify errors from versioning factory/service are caught."""
        mock_factory = MagicMock()
        mock_factory.get_model_versioning_service.side_effect = RuntimeError("Factory failed")

        persister = ParquetExperimentPersister(
            storage=storage, model_versioning_service_factory=mock_factory
        )
        model = LogisticRegression()

        # Should not raise, just log warning
        persister.save_model(model, sample_context)


class DescribeParquetExperimentPersisterCheckpointExists:
    """Tests for ParquetExperimentPersister.checkpoint_exists() method."""

    def it_returns_true_when_checkpoint_exists(
        self, tmp_path: Path, storage: StorageService
    ) -> None:
        """Verify returns True when file exists."""
        checkpoint_path = tmp_path / "checkpoint.parquet"
        checkpoint_path.touch()

        persister = ParquetExperimentPersister(storage=storage)

        assert persister.checkpoint_exists(str(checkpoint_path)) is True

    def it_returns_false_when_checkpoint_not_exists(
        self, tmp_path: Path, storage: StorageService
    ) -> None:
        """Verify returns False when file doesn't exist."""
        checkpoint_uri = str(tmp_path / "nonexistent.parquet")

        persister = ParquetExperimentPersister(storage=storage)

        assert persister.checkpoint_exists(checkpoint_uri) is False


class DescribeParquetExperimentPersisterDiscardCheckpoint:
    """Tests for ParquetExperimentPersister.discard_checkpoint() method."""

    def it_removes_existing_checkpoint(self, tmp_path: Path, storage: StorageService) -> None:
        """Verify checkpoint file is removed."""
        checkpoint_path = tmp_path / "checkpoint.parquet"
        checkpoint_path.touch()
        assert checkpoint_path.exists()

        persister = ParquetExperimentPersister(storage=storage)
        persister.discard_checkpoint(str(checkpoint_path))

        assert not checkpoint_path.exists()

    def it_handles_nonexistent_checkpoint_gracefully(
        self, tmp_path: Path, storage: StorageService
    ) -> None:
        """Verify no error when checkpoint doesn't exist."""
        checkpoint_uri = str(tmp_path / "nonexistent.parquet")

        persister = ParquetExperimentPersister(storage=storage)

        # Should not raise
        persister.discard_checkpoint(checkpoint_uri)


class DescribeCompositeExperimentPersister:
    """Tests for CompositeExperimentPersister class."""

    def it_initializes_with_list_of_persisters(self, storage: StorageService) -> None:
        """Verify composite stores list of persisters."""
        persister1 = ParquetExperimentPersister(storage=storage)
        persister2 = ParquetExperimentPersister(storage=storage)

        composite = CompositeExperimentPersister([persister1, persister2])

        assert len(composite._persisters) == 2

    def it_handles_empty_persister_list(self) -> None:
        """Verify composite handles empty list."""
        composite = CompositeExperimentPersister([])

        assert composite._persisters == []


class DescribeCompositeExperimentPersisterSaveCheckpoint:
    """Tests for CompositeExperimentPersister.save_checkpoint() method."""

    def it_calls_all_child_persisters(
        self,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Verify save_checkpoint is called on all persisters."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        composite = CompositeExperimentPersister([mock1, mock2])

        checkpoint_uri = str(tmp_path / "checkpoint.parquet")
        composite.save_checkpoint(sample_metrics, checkpoint_uri)

        mock1.save_checkpoint.assert_called_once_with(sample_metrics, checkpoint_uri)
        mock2.save_checkpoint.assert_called_once_with(sample_metrics, checkpoint_uri)


class DescribeCompositeExperimentPersisterSaveModel:
    """Tests for CompositeExperimentPersister.save_model() method."""

    def it_calls_all_child_persisters(
        self,
        sample_context: ExperimentContext,
    ) -> None:
        """Verify save_model is called on all persisters."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        composite = CompositeExperimentPersister([mock1, mock2])
        model = LogisticRegression()

        composite.save_model(model, sample_context)

        mock1.save_model.assert_called_once_with(model, sample_context)
        mock2.save_model.assert_called_once_with(model, sample_context)


class DescribeCompositeExperimentPersisterCheckpointExists:
    """Tests for CompositeExperimentPersister.checkpoint_exists() method."""

    def it_uses_first_persister_for_check(self, tmp_path: Path) -> None:
        """Verify checkpoint_exists uses first persister."""
        mock1 = MagicMock()
        mock1.checkpoint_exists.return_value = True
        mock2 = MagicMock()
        mock2.checkpoint_exists.return_value = False

        composite = CompositeExperimentPersister([mock1, mock2])

        checkpoint_uri = str(tmp_path / "checkpoint.parquet")
        result = composite.checkpoint_exists(checkpoint_uri)

        assert result is True
        mock1.checkpoint_exists.assert_called_once_with(checkpoint_uri)
        mock2.checkpoint_exists.assert_not_called()

    def it_returns_false_when_no_persisters(self, tmp_path: Path) -> None:
        """Verify returns False when persister list is empty."""
        composite = CompositeExperimentPersister([])

        checkpoint_uri = str(tmp_path / "checkpoint.parquet")
        result = composite.checkpoint_exists(checkpoint_uri)

        assert result is False


class DescribeCompositeExperimentPersisterDiscardCheckpoint:
    """Tests for CompositeExperimentPersister.discard_checkpoint() method."""

    def it_calls_all_child_persisters(self, tmp_path: Path) -> None:
        """Verify discard_checkpoint is called on all persisters."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        composite = CompositeExperimentPersister([mock1, mock2])

        checkpoint_uri = str(tmp_path / "checkpoint.parquet")
        composite.discard_checkpoint(checkpoint_uri)

        mock1.discard_checkpoint.assert_called_once_with(checkpoint_uri)
        mock2.discard_checkpoint.assert_called_once_with(checkpoint_uri)
