"""Tests for model_repository service."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.services.model_repository import (
    ModelStorageLayout,
    ModelStorageRepository,
    StoredModel,
)
from experiments.services.model_versioning import (
    ModelNotFoundError,
    ModelVersion,
    ModelVersionsQuery,
)
from experiments.storage.interface import FileInfo

# ============================================================================
# ModelStorageLayout Tests
# ============================================================================


class DescribeModelStorageLayoutDefaults:
    def it_has_default_model_key_template(self, model_layout: ModelStorageLayout) -> None:
        assert (
            "models/{dataset}/{model_type}/{technique}/{model_id}.joblib"
            in model_layout.model_key_template
        )

    def it_has_default_model_prefix(self, model_layout: ModelStorageLayout) -> None:
        assert model_layout.model_prefix == "models/"


class DescribeGetModelKey:
    def it_formats_key_with_all_components(self, model_layout: ModelStorageLayout) -> None:
        key = model_layout.get_model_key(
            dataset="taiwan_credit",
            model_type="random_forest",
            technique="smote",
            model_id="abc123",
        )

        assert key == "models/taiwan_credit/random_forest/smote/abc123.joblib"


class DescribeParseModelKey:
    def it_parses_valid_key(self, model_layout: ModelStorageLayout) -> None:
        key = "models/taiwan_credit/random_forest/smote/model123.joblib"

        result = model_layout.parse_model_key(key)

        assert result is not None
        assert result["dataset"] == "taiwan_credit"
        assert result["model_type"] == "random_forest"
        assert result["technique"] == "smote"
        assert result["model_id"] == "model123"

    def it_returns_none_for_invalid_key(self, model_layout: ModelStorageLayout) -> None:
        result = model_layout.parse_model_key("invalid/path/file.txt")

        assert result is None

    def it_returns_none_for_wrong_extension(self, model_layout: ModelStorageLayout) -> None:
        result = model_layout.parse_model_key("models/dataset/model_type/technique/model.pickle")

        assert result is None


# ============================================================================
# ModelStorageRepository Tests
# ============================================================================


class DescribeModelStorageRepositoryInit:
    def it_stores_storage_backend(self, mock_storage: MagicMock) -> None:
        repo = ModelStorageRepository(storage=mock_storage)

        assert repo._storage is mock_storage

    def it_uses_default_layout_if_not_provided(self, mock_storage: MagicMock) -> None:
        repo = ModelStorageRepository(storage=mock_storage)

        assert isinstance(repo._layout, ModelStorageLayout)

    def it_uses_provided_layout(self, mock_storage: MagicMock) -> None:
        layout = ModelStorageLayout(model_prefix="custom/")

        repo = ModelStorageRepository(storage=mock_storage, layout=layout)

        assert repo._layout is layout


class DescribeSaveModel:
    def it_writes_model_to_storage(
        self, mock_storage: MagicMock, model_repository: ModelStorageRepository, valid_uuid: str
    ) -> None:
        model = MagicMock()

        model_repository.save_model(
            model=model,
            id=valid_uuid,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            params={"n_estimators": 100},
            seed=42,
        )

        mock_storage.write_joblib.assert_called_once()
        call_args = mock_storage.write_joblib.call_args
        assert f"{valid_uuid}.joblib" in call_args[0][1]

    def it_returns_model_version(
        self, mock_storage: MagicMock, model_repository: ModelStorageRepository, valid_uuid: str
    ) -> None:
        result = model_repository.save_model(
            model=MagicMock(),
            id=valid_uuid,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            params={},
            seed=42,
        )

        assert isinstance(result, ModelVersion)
        assert str(result.id) == valid_uuid
        assert result.type == ModelType.RANDOM_FOREST
        assert result.technique == Technique.SMOTE

    def it_generates_id_if_not_provided(
        self, mock_storage: MagicMock, model_repository: ModelStorageRepository
    ) -> None:
        result = model_repository.save_model(
            model=MagicMock(),
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            params={},
            seed=42,
        )

        assert result.id is not None
        assert len(str(result.id)) > 0

    def it_stores_model_with_all_metadata(
        self, mock_storage: MagicMock, model_repository: ModelStorageRepository, valid_uuid: str
    ) -> None:
        model = MagicMock()
        params = {"C": 1.0}

        model_repository.save_model(
            model=model,
            id=valid_uuid,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            params=params,
            seed=123,
        )

        stored_model = mock_storage.write_joblib.call_args[0][0]
        assert isinstance(stored_model, StoredModel)
        assert stored_model.model is model
        assert stored_model.params == params
        assert stored_model.seed == 123


class DescribeListModels:
    def it_returns_empty_list_when_no_models(
        self, mock_storage: MagicMock, model_repository: ModelStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = []

        result = model_repository.list_models(dataset=Dataset.TAIWAN_CREDIT)

        assert result.versions == []
        assert result.total_count == 0

    def it_filters_by_dataset(
        self,
        mock_storage: MagicMock,
        model_repository: ModelStorageRepository,
        valid_uuid: str,
        another_valid_uuid: str,
        sample_timestamp: datetime,
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
            FileInfo(
                key=f"models/lending_club/random_forest/baseline/{another_valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
        ]

        result = model_repository.list_models(dataset=Dataset.TAIWAN_CREDIT)

        assert result.total_count == 1
        assert str(result.versions[0].id) == valid_uuid

    def it_filters_by_model_type(
        self,
        mock_storage: MagicMock,
        model_repository: ModelStorageRepository,
        valid_uuid: str,
        another_valid_uuid: str,
        sample_timestamp: datetime,
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
            FileInfo(
                key=f"models/taiwan_credit/svm/baseline/{another_valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
        ]
        query = ModelVersionsQuery(model_type=ModelType.RANDOM_FOREST)

        result = model_repository.list_models(dataset=Dataset.TAIWAN_CREDIT, params=query)

        assert result.total_count == 1
        assert result.versions[0].type == ModelType.RANDOM_FOREST

    def it_filters_by_technique(
        self,
        mock_storage: MagicMock,
        model_repository: ModelStorageRepository,
        valid_uuid: str,
        another_valid_uuid: str,
        sample_timestamp: datetime,
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
            FileInfo(
                key=f"models/taiwan_credit/random_forest/smote/{another_valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
        ]
        query = ModelVersionsQuery(technique=Technique.SMOTE)

        result = model_repository.list_models(dataset=Dataset.TAIWAN_CREDIT, params=query)

        assert result.total_count == 1
        assert result.versions[0].technique == Technique.SMOTE

    def it_skips_invalid_keys(
        self,
        mock_storage: MagicMock,
        model_repository: ModelStorageRepository,
        valid_uuid: str,
        sample_timestamp: datetime,
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
            FileInfo(key="invalid/path/file.txt", size_bytes=100, last_modified=sample_timestamp),
        ]

        result = model_repository.list_models(dataset=Dataset.TAIWAN_CREDIT)

        assert result.total_count == 1


class DescribeGetVersion:
    def it_returns_model_when_found(
        self,
        mock_storage: MagicMock,
        model_repository: ModelStorageRepository,
        valid_uuid: str,
        sample_timestamp: datetime,
    ) -> None:
        mock_model = MagicMock()
        stored = StoredModel(
            model=mock_model,
            params={"C": 1.0},
            seed=42,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
        )
        mock_storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/svm/baseline/{valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
        ]
        mock_storage.read_joblib.return_value = stored

        result = model_repository.get_version(dataset=Dataset.TAIWAN_CREDIT, id=valid_uuid)

        assert str(result.version.id) == valid_uuid
        assert result.model.model is mock_model
        assert result.model.params == {"C": 1.0}

    def it_raises_model_not_found_when_missing(
        self, mock_storage: MagicMock, model_repository: ModelStorageRepository
    ) -> None:
        mock_storage.list_files.return_value = []

        with pytest.raises(ModelNotFoundError):
            model_repository.get_version(dataset=Dataset.TAIWAN_CREDIT, id="nonexistent")

    def it_raises_when_id_not_matching(
        self,
        mock_storage: MagicMock,
        model_repository: ModelStorageRepository,
        another_valid_uuid: str,
        sample_timestamp: datetime,
    ) -> None:
        mock_storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/svm/baseline/{another_valid_uuid}.joblib",
                size_bytes=1024,
                last_modified=sample_timestamp,
            ),
        ]

        with pytest.raises(ModelNotFoundError):
            model_repository.get_version(dataset=Dataset.TAIWAN_CREDIT, id="nonexistent")
