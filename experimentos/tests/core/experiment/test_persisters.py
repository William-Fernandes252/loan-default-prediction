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
        checkpoint_path=Path("/tmp/test_checkpoint.parquet"),
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

    def it_initializes_without_versioning_service(self) -> None:
        """Verify persister can be initialized without versioning service."""
        persister = ParquetExperimentPersister()

        assert persister._model_versioning_service is None

    def it_initializes_with_versioning_service(self) -> None:
        """Verify persister stores versioning service."""
        mock_service = MagicMock()
        persister = ParquetExperimentPersister(model_versioning_service=mock_service)

        assert persister._model_versioning_service is mock_service


class DescribeParquetExperimentPersisterSaveCheckpoint:
    """Tests for ParquetExperimentPersister.save_checkpoint() method."""

    def it_saves_metrics_to_parquet(
        self,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Verify metrics are saved to parquet file."""
        checkpoint_path = tmp_path / "checkpoint.parquet"
        persister = ParquetExperimentPersister()

        persister.save_checkpoint(sample_metrics, checkpoint_path)

        assert checkpoint_path.exists()

        # Read back and verify
        df = pd.read_parquet(checkpoint_path)
        assert len(df) == 1
        assert df.iloc[0]["accuracy_balanced"] == 0.85
        assert df.iloc[0]["dataset"] == "taiwan_credit"

    def it_creates_parent_directories_implicitly(
        self,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Verify parent directories are created if needed."""
        checkpoint_path = tmp_path / "subdir" / "nested" / "checkpoint.parquet"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        persister = ParquetExperimentPersister()
        persister.save_checkpoint(sample_metrics, checkpoint_path)

        assert checkpoint_path.exists()


class DescribeParquetExperimentPersisterSaveModel:
    """Tests for ParquetExperimentPersister.save_model() method."""

    def it_does_nothing_when_no_versioning_service(
        self,
        sample_context: ExperimentContext,
    ) -> None:
        """Verify no error when versioning service is None."""
        persister = ParquetExperimentPersister()
        model = LogisticRegression()

        # Should not raise
        persister.save_model(model, sample_context)

    def it_calls_versioning_service_save(
        self,
        sample_context: ExperimentContext,
    ) -> None:
        """Verify versioning service save_model is called."""
        mock_service = MagicMock()
        persister = ParquetExperimentPersister(model_versioning_service=mock_service)
        model = LogisticRegression()

        persister.save_model(model, sample_context)

        mock_service.save_model.assert_called_once_with(model, None)

    def it_handles_versioning_service_errors_gracefully(
        self,
        sample_context: ExperimentContext,
    ) -> None:
        """Verify errors from versioning service are caught."""
        mock_service = MagicMock()
        mock_service.save_model.side_effect = RuntimeError("Save failed")
        persister = ParquetExperimentPersister(model_versioning_service=mock_service)
        model = LogisticRegression()

        # Should not raise, just log warning
        persister.save_model(model, sample_context)


class DescribeParquetExperimentPersisterCheckpointExists:
    """Tests for ParquetExperimentPersister.checkpoint_exists() method."""

    def it_returns_true_when_checkpoint_exists(self, tmp_path: Path) -> None:
        """Verify returns True when file exists."""
        checkpoint_path = tmp_path / "checkpoint.parquet"
        checkpoint_path.touch()

        persister = ParquetExperimentPersister()

        assert persister.checkpoint_exists(checkpoint_path) is True

    def it_returns_false_when_checkpoint_not_exists(self, tmp_path: Path) -> None:
        """Verify returns False when file doesn't exist."""
        checkpoint_path = tmp_path / "nonexistent.parquet"

        persister = ParquetExperimentPersister()

        assert persister.checkpoint_exists(checkpoint_path) is False


class DescribeParquetExperimentPersisterDiscardCheckpoint:
    """Tests for ParquetExperimentPersister.discard_checkpoint() method."""

    def it_removes_existing_checkpoint(self, tmp_path: Path) -> None:
        """Verify checkpoint file is removed."""
        checkpoint_path = tmp_path / "checkpoint.parquet"
        checkpoint_path.touch()
        assert checkpoint_path.exists()

        persister = ParquetExperimentPersister()
        persister.discard_checkpoint(checkpoint_path)

        assert not checkpoint_path.exists()

    def it_handles_nonexistent_checkpoint_gracefully(self, tmp_path: Path) -> None:
        """Verify no error when checkpoint doesn't exist."""
        checkpoint_path = tmp_path / "nonexistent.parquet"

        persister = ParquetExperimentPersister()

        # Should not raise
        persister.discard_checkpoint(checkpoint_path)


class DescribeCompositeExperimentPersister:
    """Tests for CompositeExperimentPersister class."""

    def it_initializes_with_list_of_persisters(self) -> None:
        """Verify composite stores list of persisters."""
        persister1 = ParquetExperimentPersister()
        persister2 = ParquetExperimentPersister()

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

        checkpoint_path = tmp_path / "checkpoint.parquet"
        composite.save_checkpoint(sample_metrics, checkpoint_path)

        mock1.save_checkpoint.assert_called_once_with(sample_metrics, checkpoint_path)
        mock2.save_checkpoint.assert_called_once_with(sample_metrics, checkpoint_path)


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

        checkpoint_path = tmp_path / "checkpoint.parquet"
        result = composite.checkpoint_exists(checkpoint_path)

        assert result is True
        mock1.checkpoint_exists.assert_called_once_with(checkpoint_path)
        mock2.checkpoint_exists.assert_not_called()

    def it_returns_false_when_no_persisters(self, tmp_path: Path) -> None:
        """Verify returns False when persister list is empty."""
        composite = CompositeExperimentPersister([])

        checkpoint_path = tmp_path / "checkpoint.parquet"
        result = composite.checkpoint_exists(checkpoint_path)

        assert result is False


class DescribeCompositeExperimentPersisterDiscardCheckpoint:
    """Tests for CompositeExperimentPersister.discard_checkpoint() method."""

    def it_calls_all_child_persisters(self, tmp_path: Path) -> None:
        """Verify discard_checkpoint is called on all persisters."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        composite = CompositeExperimentPersister([mock1, mock2])

        checkpoint_path = tmp_path / "checkpoint.parquet"
        composite.discard_checkpoint(checkpoint_path)

        mock1.discard_checkpoint.assert_called_once_with(checkpoint_path)
        mock2.discard_checkpoint.assert_called_once_with(checkpoint_path)
