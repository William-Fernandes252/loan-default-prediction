"""Tests for model_predictions_repository service."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import RawPredictions
from experiments.services.model_predictions_repository import (
    ModelPredictionsStorageLayout,
    ModelPredictionsStorageRepository,
)
from experiments.storage.interface import FileInfo


class DescribeModelPredictionsStorageLayoutDefaults:
    def it_has_default_predictions_key_template(self) -> None:
        layout = ModelPredictionsStorageLayout()

        assert "predictions/{execution_id}" in layout.predictions_key_template

    def it_has_default_predictions_prefix(self) -> None:
        layout = ModelPredictionsStorageLayout()

        assert layout.predictions_prefix == "predictions/"


class DescribeGetPredictionsKey:
    def it_formats_key_with_all_components(self) -> None:
        layout = ModelPredictionsStorageLayout()

        key = layout.get_predictions_key(
            execution_id="exec-123",
            dataset="taiwan_credit",
            model_type="random_forest",
            technique="smote",
            seed=42,
        )

        expected = "predictions/exec-123/taiwan_credit/random_forest/smote/seed_42.parquet"
        assert key == expected


class DescribeParsePredictionsKey:
    @pytest.fixture
    def layout(self) -> ModelPredictionsStorageLayout:
        return ModelPredictionsStorageLayout()

    def it_parses_valid_key(self, layout: ModelPredictionsStorageLayout) -> None:
        key = "predictions/exec-123/taiwan_credit/random_forest/smote/seed_42.parquet"

        result = layout.parse_predictions_key(key)

        assert result is not None
        assert result["execution_id"] == "exec-123"
        assert result["dataset"] == "taiwan_credit"
        assert result["model_type"] == "random_forest"
        assert result["technique"] == "smote"
        assert result["seed"] == 42

    def it_returns_none_for_invalid_key(self, layout: ModelPredictionsStorageLayout) -> None:
        key = "invalid/path/file.txt"

        result = layout.parse_predictions_key(key)

        assert result is None

    def it_returns_none_for_wrong_extension(self, layout: ModelPredictionsStorageLayout) -> None:
        key = "predictions/exec-123/dataset/model/technique/seed_42.csv"

        result = layout.parse_predictions_key(key)

        assert result is None


class DescribeModelPredictionsStorageRepositoryInit:
    def it_stores_storage_backend(self) -> None:
        storage = MagicMock()

        repo = ModelPredictionsStorageRepository(storage=storage)

        assert repo._storage is storage

    def it_uses_default_layout_if_not_provided(self) -> None:
        storage = MagicMock()

        repo = ModelPredictionsStorageRepository(storage=storage)

        assert isinstance(repo._layout, ModelPredictionsStorageLayout)

    def it_uses_provided_layout(self) -> None:
        storage = MagicMock()
        layout = ModelPredictionsStorageLayout(predictions_prefix="custom/")

        repo = ModelPredictionsStorageRepository(storage=storage, layout=layout)

        assert repo._layout is layout


class DescribeGetCompletedCombinations:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> ModelPredictionsStorageRepository:
        return ModelPredictionsStorageRepository(storage=storage)

    def it_returns_empty_set_when_no_files(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        storage.list_files.return_value = []

        result = repo.get_completed_combinations(execution_id="exec-123")

        assert result == set()

    def it_returns_combinations_for_execution(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = repo.get_completed_combinations(execution_id="exec-123")

        assert len(result) == 1
        combo = next(iter(result))
        assert combo.dataset == Dataset.TAIWAN_CREDIT
        assert combo.model_type == ModelType.RANDOM_FOREST
        assert combo.technique == Technique.BASELINE
        assert combo.seed == 42

    def it_filters_by_execution_id(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-other/taiwan_credit/svm/smote/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = repo.get_completed_combinations(execution_id="exec-123")

        assert len(result) == 1

    def it_skips_invalid_enum_values(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-123/invalid_dataset/invalid_model/invalid/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = repo.get_completed_combinations(execution_id="exec-123")

        # Only valid combination should be returned
        assert len(result) == 1


class DescribeSavePredictions:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> ModelPredictionsStorageRepository:
        return ModelPredictionsStorageRepository(storage=storage)

    def it_writes_predictions_to_storage(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        predictions = RawPredictions(
            target=np.array([0, 1, 0]),
            prediction=np.array([0, 1, 1]),
        )

        repo.save_predictions(
            execution_id="exec-123",
            seed=42,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            predictions=predictions,
        )

        storage.sink_parquet.assert_called_once()

    def it_uses_correct_storage_key(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        predictions = RawPredictions(
            target=np.array([0, 1]),
            prediction=np.array([0, 1]),
        )

        repo.save_predictions(
            execution_id="exec-456",
            seed=100,
            dataset=Dataset.LENDING_CLUB,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            predictions=predictions,
        )

        call_args = storage.sink_parquet.call_args
        key = call_args[0][1]
        assert "exec-456" in key
        assert "lending_club" in key
        assert "svm" in key
        assert "baseline" in key
        assert "seed_100" in key


class DescribeGetLatestPredictionsForExperiment:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> ModelPredictionsStorageRepository:
        return ModelPredictionsStorageRepository(storage=storage)

    def it_returns_none_when_no_files(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        storage.list_files.return_value = []

        result = repo.get_latest_predictions_for_experiment(dataset=Dataset.TAIWAN_CREDIT)

        assert result is None

    def it_returns_none_when_no_matching_dataset(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-123/lending_club/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]

        result = repo.get_latest_predictions_for_experiment(dataset=Dataset.TAIWAN_CREDIT)

        assert result is None

    def it_returns_iterator_of_predictions_for_latest_execution(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        mock_lazy_frame = MagicMock()
        storage.list_files.return_value = [
            FileInfo(
                key="predictions/exec-001/taiwan_credit/random_forest/baseline/seed_42.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/exec-002/taiwan_credit/svm/smote/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        storage.scan_parquet.return_value = mock_lazy_frame

        result = repo.get_latest_predictions_for_experiment(dataset=Dataset.TAIWAN_CREDIT)

        # Result should be an iterator, consume it
        predictions_list = list(result) if result else []

        # Should return only predictions from exec-002 (latest)
        assert len(predictions_list) == 1
        assert predictions_list[0].execution_id == "exec-002"

    def it_uses_uuid7_sortable_execution_ids(
        self, storage: MagicMock, repo: ModelPredictionsStorageRepository
    ) -> None:
        mock_lazy_frame = MagicMock()
        storage.list_files.return_value = [
            FileInfo(
                key="predictions/aaa-exec/taiwan_credit/random_forest/baseline/seed_1.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
            FileInfo(
                key="predictions/zzz-exec/taiwan_credit/svm/baseline/seed_2.parquet",
                size_bytes=1024,
                last_modified=None,
            ),
        ]
        storage.scan_parquet.return_value = mock_lazy_frame

        result = repo.get_latest_predictions_for_experiment(dataset=Dataset.TAIWAN_CREDIT)
        predictions_list = list(result) if result else []

        # Latest execution_id by sorting is "zzz-exec"
        assert len(predictions_list) == 1
        assert predictions_list[0].execution_id == "zzz-exec"
