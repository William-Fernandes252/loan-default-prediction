"""Shared fixtures for services tests."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import ModelPredictions
from experiments.core.training.data import TrainingData
from experiments.services.data_repository import DataStorageLayout, StorageDataRepository
from experiments.services.feature_extractor import FeatureExtractorImpl
from experiments.services.grid_search_trainer import GridSearchModelTrainer
from experiments.services.model_predictions_repository import (
    ModelPredictionsStorageLayout,
    ModelPredictionsStorageRepository,
)
from experiments.services.model_repository import ModelStorageLayout, ModelStorageRepository
from experiments.services.model_results_evaluator import ModelResultsEvaluatorImpl
from experiments.services.resource_calculator import ResourceCalculator
from experiments.services.stratified_data_splitter import StratifiedDataSplitter
from experiments.services.unbalanced_learner_factory import UnbalancedLearnerFactory
from experiments.storage.interface import FileInfo

# ============================================================================
# Common Test Data
# ============================================================================


@pytest.fixture
def valid_uuid() -> str:
    """A valid UUID v7 format string for testing."""
    return "01912345-6789-7abc-8def-0123456789ab"


@pytest.fixture
def another_valid_uuid() -> str:
    """A second valid UUID for testing multiple models."""
    return "01912345-6789-7abc-8def-0123456789cd"


@pytest.fixture
def sample_timestamp() -> datetime:
    """A fixed timestamp for deterministic tests."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# Mock Storage Backend
# ============================================================================


@pytest.fixture
def mock_storage() -> MagicMock:
    """A mock storage backend with no pre-configured behavior."""
    return MagicMock()


# ============================================================================
# Data Repository Fixtures
# ============================================================================


@pytest.fixture
def data_layout() -> DataStorageLayout:
    """Default data storage layout."""
    return DataStorageLayout()


@pytest.fixture
def data_repository(
    mock_storage: MagicMock, data_layout: DataStorageLayout
) -> StorageDataRepository:
    """Data repository with mock storage."""
    return StorageDataRepository(mock_storage, data_layout)


# ============================================================================
# Model Repository Fixtures
# ============================================================================


@pytest.fixture
def model_layout() -> ModelStorageLayout:
    """Default model storage layout."""
    return ModelStorageLayout()


@pytest.fixture
def model_repository(mock_storage: MagicMock) -> ModelStorageRepository:
    """Model repository with mock storage."""
    return ModelStorageRepository(storage=mock_storage)


# ============================================================================
# Predictions Repository Fixtures
# ============================================================================


@pytest.fixture
def predictions_layout() -> ModelPredictionsStorageLayout:
    """Default predictions storage layout."""
    return ModelPredictionsStorageLayout()


@pytest.fixture
def predictions_repository(mock_storage: MagicMock) -> ModelPredictionsStorageRepository:
    """Predictions repository with mock storage."""
    return ModelPredictionsStorageRepository(storage=mock_storage)


# ============================================================================
# ML Service Fixtures
# ============================================================================


@pytest.fixture
def feature_extractor() -> FeatureExtractorImpl:
    """Feature extractor instance."""
    return FeatureExtractorImpl()


@pytest.fixture
def resource_calculator() -> ResourceCalculator:
    """Resource calculator with default settings."""
    return ResourceCalculator(safety_factor=3.5)


@pytest.fixture
def stratified_splitter() -> StratifiedDataSplitter:
    """Stratified data splitter with test-friendly settings."""
    return StratifiedDataSplitter(test_size=0.30, cv_folds=2)


@pytest.fixture
def learner_factory() -> UnbalancedLearnerFactory:
    """Unbalanced learner factory without GPU."""
    return UnbalancedLearnerFactory(use_gpu=False)


@pytest.fixture
def grid_search_trainer() -> GridSearchModelTrainer:
    """Grid search trainer with default settings."""
    return GridSearchModelTrainer(cost_grids=[])


@pytest.fixture
def results_evaluator() -> ModelResultsEvaluatorImpl:
    """Model results evaluator instance."""
    return ModelResultsEvaluatorImpl()


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """A simple DataFrame for testing."""
    return pl.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_lazyframe(sample_dataframe: pl.DataFrame) -> pl.LazyFrame:
    """A simple LazyFrame for testing."""
    return sample_dataframe.lazy()


@pytest.fixture
def balanced_training_data() -> TrainingData:
    """Balanced training data with 50 samples per class."""
    n_samples = 100
    X = pl.DataFrame(
        {
            "feature1": list(range(n_samples)),
            "feature2": list(range(n_samples, n_samples * 2)),
        }
    )
    y = pl.DataFrame({"target": [0] * 50 + [1] * 50})
    return TrainingData(X=X, y=y)


@pytest.fixture
def imbalanced_training_data() -> TrainingData:
    """Imbalanced training data (80% class 0, 20% class 1)."""
    n_samples = 50
    X = pl.DataFrame({"feature": list(range(n_samples))})
    y = pl.DataFrame({"target": [0] * 40 + [1] * 10})
    return TrainingData(X=X, y=y)


@pytest.fixture
def sample_predictions_array() -> tuple[np.ndarray, np.ndarray]:
    """Sample target and prediction arrays."""
    target = np.array([0, 1, 0, 1])
    prediction = np.array([0, 1, 1, 1])
    return target, prediction


@pytest.fixture
def perfect_predictions_lf() -> pl.LazyFrame:
    """LazyFrame with perfect predictions (100% accuracy)."""
    return pl.LazyFrame(
        {
            "target": [1, 1, 0, 0],
            "prediction": [1, 1, 0, 0],
        }
    )


@pytest.fixture
def mixed_predictions_lf() -> pl.LazyFrame:
    """LazyFrame with 50% accuracy (TP=1, TN=1, FP=1, FN=1)."""
    return pl.LazyFrame(
        {
            "target": [1, 1, 0, 0],
            "prediction": [1, 0, 0, 1],
        }
    )


# ============================================================================
# File Info Fixtures
# ============================================================================


def make_file_info(
    key: str, size_bytes: int = 1024, last_modified: datetime | None = None
) -> FileInfo:
    """Factory function for creating FileInfo instances."""
    return FileInfo(key=key, size_bytes=size_bytes, last_modified=last_modified)


@pytest.fixture
def make_model_file_info(sample_timestamp: datetime):
    """Factory fixture for creating model FileInfo with standard format."""

    def _factory(
        dataset: str = "taiwan_credit",
        model_type: str = "random_forest",
        technique: str = "baseline",
        model_id: str = "01912345-6789-7abc-8def-0123456789ab",
    ) -> FileInfo:
        key = f"models/{dataset}/{model_type}/{technique}/{model_id}.joblib"
        return FileInfo(key=key, size_bytes=1024, last_modified=sample_timestamp)

    return _factory


@pytest.fixture
def make_predictions_file_info():
    """Factory fixture for creating predictions FileInfo with standard format."""

    def _factory(
        execution_id: str = "exec-123",
        dataset: str = "taiwan_credit",
        model_type: str = "random_forest",
        technique: str = "baseline",
        seed: int = 42,
    ) -> FileInfo:
        key = f"predictions/{execution_id}/{dataset}/{model_type}/{technique}/seed_{seed}.parquet"
        return FileInfo(key=key, size_bytes=1024, last_modified=None)

    return _factory


# ============================================================================
# Model Predictions Fixtures
# ============================================================================


def make_model_predictions(
    execution_id: str = "exec-1",
    seed: int = 42,
    model_type: ModelType = ModelType.RANDOM_FOREST,
    technique: Technique = Technique.BASELINE,
    target: list[int] | None = None,
    prediction: list[int] | None = None,
) -> ModelPredictions:
    """Factory function for creating ModelPredictions instances."""
    if target is None:
        target = [1, 0]
    if prediction is None:
        prediction = [1, 0]

    return ModelPredictions(
        execution_id=execution_id,
        seed=seed,
        dataset=MagicMock(),
        model_type=model_type,
        technique=technique,
        predictions=pl.LazyFrame({"target": target, "prediction": prediction}),
    )
