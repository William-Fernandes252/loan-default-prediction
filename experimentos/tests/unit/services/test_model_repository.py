"""Tests for model_repository service."""

from datetime import datetime, timezone
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


class DescribeModelStorageLayoutDefaults:
    def it_has_default_model_key_template(self) -> None:
        layout = ModelStorageLayout()

        assert (
            "models/{dataset}/{model_type}/{technique}/{model_id}.joblib"
            in layout.model_key_template
        )

    def it_has_default_model_prefix(self) -> None:
        layout = ModelStorageLayout()

        assert layout.model_prefix == "models/"


class DescribeGetModelKey:
    def it_formats_key_with_all_components(self) -> None:
        layout = ModelStorageLayout()

        key = layout.get_model_key(
            dataset="taiwan_credit",
            model_type="random_forest",
            technique="smote",
            model_id="abc123",
        )

        assert key == "models/taiwan_credit/random_forest/smote/abc123.joblib"


class DescribeParseModelKey:
    @pytest.fixture
    def layout(self) -> ModelStorageLayout:
        return ModelStorageLayout()

    def it_parses_valid_key(self, layout: ModelStorageLayout) -> None:
        key = "models/taiwan_credit/random_forest/smote/model123.joblib"

        result = layout.parse_model_key(key)

        assert result is not None
        assert result["dataset"] == "taiwan_credit"
        assert result["model_type"] == "random_forest"
        assert result["technique"] == "smote"
        assert result["model_id"] == "model123"

    def it_returns_none_for_invalid_key(self, layout: ModelStorageLayout) -> None:
        key = "invalid/path/file.txt"

        result = layout.parse_model_key(key)

        assert result is None

    def it_returns_none_for_wrong_extension(self, layout: ModelStorageLayout) -> None:
        key = "models/dataset/model_type/technique/model.pickle"

        result = layout.parse_model_key(key)

        assert result is None


class DescribeModelStorageRepositoryInit:
    def it_stores_storage_backend(self) -> None:
        storage = MagicMock()

        repo = ModelStorageRepository(storage=storage)

        assert repo._storage is storage

    def it_uses_default_layout_if_not_provided(self) -> None:
        storage = MagicMock()

        repo = ModelStorageRepository(storage=storage)

        assert isinstance(repo._layout, ModelStorageLayout)

    def it_uses_provided_layout(self) -> None:
        storage = MagicMock()
        layout = ModelStorageLayout(model_prefix="custom/")

        repo = ModelStorageRepository(storage=storage, layout=layout)

        assert repo._layout is layout


class DescribeSaveModel:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> ModelStorageRepository:
        return ModelStorageRepository(storage=storage)

    @pytest.fixture
    def test_uuid(self) -> str:
        return "01912345-6789-7abc-8def-0123456789ab"

    def it_writes_model_to_storage(
        self, storage: MagicMock, repo: ModelStorageRepository, test_uuid: str
    ) -> None:
        model = MagicMock()

        repo.save_model(
            model=model,
            id=test_uuid,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            params={"n_estimators": 100},
            seed=42,
        )

        storage.write_joblib.assert_called_once()
        call_args = storage.write_joblib.call_args
        assert f"{test_uuid}.joblib" in call_args[0][1]

    def it_returns_model_version(
        self, storage: MagicMock, repo: ModelStorageRepository, test_uuid: str
    ) -> None:
        model = MagicMock()

        result = repo.save_model(
            model=model,
            id=test_uuid,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            params={},
            seed=42,
        )

        assert isinstance(result, ModelVersion)
        assert str(result.id) == test_uuid
        assert result.type == ModelType.RANDOM_FOREST
        assert result.technique == Technique.SMOTE

    def it_generates_id_if_not_provided(
        self, storage: MagicMock, repo: ModelStorageRepository
    ) -> None:
        model = MagicMock()

        result = repo.save_model(
            model=model,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            params={},
            seed=42,
        )

        assert result.id is not None
        assert len(str(result.id)) > 0

    def it_stores_model_with_all_metadata(
        self, storage: MagicMock, repo: ModelStorageRepository, test_uuid: str
    ) -> None:
        model = MagicMock()
        params = {"C": 1.0}

        repo.save_model(
            model=model,
            id=test_uuid,
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            params=params,
            seed=123,
        )

        stored_model = storage.write_joblib.call_args[0][0]
        assert isinstance(stored_model, StoredModel)
        assert stored_model.model is model
        assert stored_model.params == params
        assert stored_model.seed == 123


class DescribeListModels:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> ModelStorageRepository:
        return ModelStorageRepository(storage=storage)

    @pytest.fixture
    def uuid1(self) -> str:
        return "01912345-6789-7abc-8def-0123456789a1"

    @pytest.fixture
    def uuid2(self) -> str:
        return "01912345-6789-7abc-8def-0123456789a2"

    def it_returns_empty_list_when_no_models(
        self, storage: MagicMock, repo: ModelStorageRepository
    ) -> None:
        storage.list_files.return_value = []

        result = repo.list_models(dataset=Dataset.TAIWAN_CREDIT)

        assert result.versions == []
        assert result.total_count == 0

    def it_filters_by_dataset(
        self, storage: MagicMock, repo: ModelStorageRepository, uuid1: str, uuid2: str
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{uuid1}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
            FileInfo(
                key=f"models/lending_club/random_forest/baseline/{uuid2}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
        ]

        result = repo.list_models(dataset=Dataset.TAIWAN_CREDIT)

        assert result.total_count == 1
        assert str(result.versions[0].id) == uuid1

    def it_filters_by_model_type_when_provided(
        self, storage: MagicMock, repo: ModelStorageRepository, uuid1: str, uuid2: str
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{uuid1}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
            FileInfo(
                key=f"models/taiwan_credit/svm/baseline/{uuid2}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
        ]
        query = ModelVersionsQuery(model_type=ModelType.RANDOM_FOREST)

        result = repo.list_models(dataset=Dataset.TAIWAN_CREDIT, params=query)

        assert result.total_count == 1
        assert result.versions[0].type == ModelType.RANDOM_FOREST

    def it_filters_by_technique_when_provided(
        self, storage: MagicMock, repo: ModelStorageRepository, uuid1: str, uuid2: str
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{uuid1}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
            FileInfo(
                key=f"models/taiwan_credit/random_forest/smote/{uuid2}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
        ]
        query = ModelVersionsQuery(technique=Technique.SMOTE)

        result = repo.list_models(dataset=Dataset.TAIWAN_CREDIT, params=query)

        assert result.total_count == 1
        assert result.versions[0].technique == Technique.SMOTE

    def it_skips_invalid_keys(
        self, storage: MagicMock, repo: ModelStorageRepository, uuid1: str
    ) -> None:
        storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/random_forest/baseline/{uuid1}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
            FileInfo(
                key="invalid/path/file.txt",
                size_bytes=100,
                last_modified=datetime.now(timezone.utc),
            ),
        ]

        result = repo.list_models(dataset=Dataset.TAIWAN_CREDIT)

        assert result.total_count == 1


class DescribeGetVersion:
    @pytest.fixture
    def storage(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, storage: MagicMock) -> ModelStorageRepository:
        return ModelStorageRepository(storage=storage)

    @pytest.fixture
    def test_uuid(self) -> str:
        return "01912345-6789-7abc-8def-0123456789ab"

    def it_returns_model_when_found(
        self, storage: MagicMock, repo: ModelStorageRepository, test_uuid: str
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
        storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/svm/baseline/{test_uuid}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
        ]
        storage.read_joblib.return_value = stored

        result = repo.get_version(dataset=Dataset.TAIWAN_CREDIT, id=test_uuid)

        assert str(result.version.id) == test_uuid
        assert result.model.model is mock_model
        assert result.model.params == {"C": 1.0}

    def it_raises_model_not_found_error_when_not_found(
        self, storage: MagicMock, repo: ModelStorageRepository
    ) -> None:
        storage.list_files.return_value = []

        with pytest.raises(ModelNotFoundError):
            repo.get_version(dataset=Dataset.TAIWAN_CREDIT, id="nonexistent")

    def it_raises_when_id_not_matching(
        self, storage: MagicMock, repo: ModelStorageRepository
    ) -> None:
        other_uuid = "01912345-6789-7abc-8def-0123456789ff"
        storage.list_files.return_value = [
            FileInfo(
                key=f"models/taiwan_credit/svm/baseline/{other_uuid}.joblib",
                size_bytes=1024,
                last_modified=datetime.now(timezone.utc),
            ),
        ]

        with pytest.raises(ModelNotFoundError):
            repo.get_version(dataset=Dataset.TAIWAN_CREDIT, id="nonexistent")
