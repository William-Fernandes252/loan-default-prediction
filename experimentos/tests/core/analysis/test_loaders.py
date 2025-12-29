"""Tests for the analysis loaders module."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.core.analysis.loaders import (
    DisplayColumnEnricher,
    EnrichedResultsLoader,
    ParquetResultsLoader,
    ResultsPathProvider,
)
from experiments.core.data import Dataset
from experiments.services.storage import StorageService
from experiments.services.storage.local import LocalStorageService


def identity_translate(s: str) -> str:
    """Identity translation function for testing."""
    return s


class FakePathProvider:
    """Fake implementation of ResultsPathProvider for testing."""

    def __init__(self, uri: str | None = None):
        self._uri = uri

    def get_latest_consolidated_results_uri(self, dataset_id: str) -> str | None:
        return self._uri


class DescribeParquetResultsLoader:
    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample results DataFrame."""
        return pd.DataFrame(
            {
                "model": ["random_forest", "svm", "xgboost"],
                "technique": ["baseline", "smote", "meta_cost"],
                "accuracy_balanced": [0.85, 0.82, 0.88],
                "f1_score": [0.75, 0.72, 0.78],
                "seed": [0, 0, 0],
            }
        )

    @pytest.fixture
    def results_path(self, tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
        """Create a temporary parquet file with sample data."""
        path = tmp_path / "results.parquet"
        sample_dataframe.to_parquet(path)
        return path

    @pytest.fixture
    def storage(self) -> StorageService:
        """Create a local storage service for testing."""
        return LocalStorageService()

    def it_satisfies_results_path_provider_protocol(self):
        """Ensure FakePathProvider satisfies the protocol."""
        provider = FakePathProvider()
        assert isinstance(provider, ResultsPathProvider)

    def it_loads_dataframe_from_parquet(self, results_path: Path, storage: StorageService):
        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(str(results_path)),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert not result.empty
        assert len(result) == 3
        assert "accuracy_balanced" in result.columns

    def it_returns_empty_dataframe_when_path_is_none(self, storage: StorageService):
        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(None),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.empty

    def it_returns_empty_dataframe_when_file_not_found(
        self, tmp_path: Path, storage: StorageService
    ):
        nonexistent_path = tmp_path / "nonexistent.parquet"
        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(str(nonexistent_path)),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.empty

    def it_does_not_add_display_columns(self, results_path: Path, storage: StorageService):
        """Verify that ParquetResultsLoader does not add display columns."""
        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(str(results_path)),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert "model_display" not in result.columns
        assert "technique_display" not in result.columns

    def it_preserves_original_columns(self, results_path: Path, storage: StorageService):
        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(str(results_path)),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert "model" in result.columns
        assert "technique" in result.columns
        assert "accuracy_balanced" in result.columns
        assert "f1_score" in result.columns
        assert "seed" in result.columns

    def it_handles_empty_parquet_file(self, tmp_path: Path, storage: StorageService):
        empty_path = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(empty_path)

        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(str(empty_path)),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.empty

    def it_handles_corrupted_file_gracefully(self, tmp_path: Path, storage: StorageService):
        """Verify that corrupted files return empty DataFrame."""
        corrupted_path = tmp_path / "corrupted.parquet"
        # Create a corrupted file (not a valid parquet file)
        corrupted_path.write_text("This is not a valid parquet file")

        loader = ParquetResultsLoader(
            path_provider=FakePathProvider(str(corrupted_path)),
            storage=storage,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.empty


class DescribeDisplayColumnEnricher:
    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample results DataFrame."""
        return pd.DataFrame(
            {
                "model": ["random_forest", "svm", "xgboost"],
                "technique": ["baseline", "smote", "meta_cost"],
                "accuracy_balanced": [0.85, 0.82, 0.88],
                "f1_score": [0.75, 0.72, 0.78],
                "seed": [0, 0, 0],
            }
        )

    def it_adds_model_display_column(self, sample_dataframe: pd.DataFrame):
        enricher = DisplayColumnEnricher(translate=identity_translate)

        result = enricher.enrich(sample_dataframe)

        assert "model_display" in result.columns
        assert result["model_display"].iloc[0] is not None

    def it_adds_technique_display_column(self, sample_dataframe: pd.DataFrame):
        enricher = DisplayColumnEnricher(translate=identity_translate)

        result = enricher.enrich(sample_dataframe)

        assert "technique_display" in result.columns
        assert result["technique_display"].iloc[0] is not None

    def it_applies_translation_function(self, sample_dataframe: pd.DataFrame):
        def bracket_translate(s: str) -> str:
            return f"[{s}]"

        enricher = DisplayColumnEnricher(translate=bracket_translate)

        result = enricher.enrich(sample_dataframe)

        # Model display should include brackets from translation
        assert result["model_display"].iloc[0].startswith("[")

    def it_preserves_original_columns(self, sample_dataframe: pd.DataFrame):
        enricher = DisplayColumnEnricher(translate=identity_translate)

        result = enricher.enrich(sample_dataframe)

        assert "model" in result.columns
        assert "technique" in result.columns
        assert "accuracy_balanced" in result.columns
        assert "f1_score" in result.columns
        assert "seed" in result.columns

    def it_handles_empty_dataframe(self):
        enricher = DisplayColumnEnricher(translate=identity_translate)

        result = enricher.enrich(pd.DataFrame())

        assert result.empty

    def it_handles_dataframe_without_model_column(self):
        df = pd.DataFrame({"accuracy_balanced": [0.85, 0.82]})
        enricher = DisplayColumnEnricher(translate=identity_translate)

        result = enricher.enrich(df)

        assert "model_display" not in result.columns
        assert "accuracy_balanced" in result.columns

    def it_handles_dataframe_without_technique_column(self):
        df = pd.DataFrame({"model": ["rf"], "accuracy_balanced": [0.85]})
        enricher = DisplayColumnEnricher(translate=identity_translate)

        result = enricher.enrich(df)

        assert "model_display" in result.columns
        assert "technique_display" not in result.columns

    def it_does_not_modify_original_dataframe(self, sample_dataframe: pd.DataFrame):
        enricher = DisplayColumnEnricher(translate=identity_translate)
        original_columns = sample_dataframe.columns.tolist()

        enricher.enrich(sample_dataframe)

        # Original dataframe should not be modified
        assert sample_dataframe.columns.tolist() == original_columns
        assert "model_display" not in sample_dataframe.columns


class DescribeEnrichedResultsLoader:
    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample results DataFrame."""
        return pd.DataFrame(
            {
                "model": ["random_forest", "svm", "xgboost"],
                "technique": ["baseline", "smote", "meta_cost"],
                "accuracy_balanced": [0.85, 0.82, 0.88],
                "f1_score": [0.75, 0.72, 0.78],
                "seed": [0, 0, 0],
            }
        )

    @pytest.fixture
    def results_path(self, tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
        """Create a temporary parquet file with sample data."""
        path = tmp_path / "results.parquet"
        sample_dataframe.to_parquet(path)
        return path

    @pytest.fixture
    def storage(self) -> StorageService:
        """Create a local storage service for testing."""
        return LocalStorageService()

    def it_loads_and_enriches_dataframe(self, results_path: Path, storage: StorageService):
        loader = EnrichedResultsLoader(
            path_provider=FakePathProvider(str(results_path)),
            storage=storage,
            translate=identity_translate,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert not result.empty
        assert len(result) == 3
        assert "accuracy_balanced" in result.columns
        assert "model_display" in result.columns
        assert "technique_display" in result.columns

    def it_returns_empty_dataframe_when_path_is_none(self, storage: StorageService):
        loader = EnrichedResultsLoader(
            path_provider=FakePathProvider(None),
            storage=storage,
            translate=identity_translate,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.empty

    def it_applies_translation_function(self, results_path: Path, storage: StorageService):
        def bracket_translate(s: str) -> str:
            return f"[{s}]"

        loader = EnrichedResultsLoader(
            path_provider=FakePathProvider(str(results_path)),
            storage=storage,
            translate=bracket_translate,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        # Model display should include brackets from translation
        assert result["model_display"].iloc[0].startswith("[")

    def it_handles_empty_parquet_file(self, tmp_path: Path, storage: StorageService):
        empty_path = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(empty_path)

        loader = EnrichedResultsLoader(
            path_provider=FakePathProvider(str(empty_path)),
            storage=storage,
            translate=identity_translate,
        )

        result = loader.load(Dataset.TAIWAN_CREDIT)

        assert result.empty
