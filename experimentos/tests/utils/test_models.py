import pytest
from sklearn.ensemble import RandomForestClassifier

from experiments.core.modeling import ModelType, Technique
from experiments.utils.models import (
    FileSystemModelRepository,
    ModelRepository,
    ModelVersioningServiceImpl,
)


class DescribeFileSystemModelRepository:
    @pytest.fixture
    def base_path(self, tmp_path):
        return tmp_path / "models"

    def it_saves_and_loads_model(self, base_path):
        repo = FileSystemModelRepository(
            base_path=base_path,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
        )
        model = RandomForestClassifier(n_estimators=10, random_state=0)

        version = repo.save_model(model, id=None)

        assert version.id is not None
        assert version.type is ModelType.RANDOM_FOREST
        assert version.technique is Technique.BASELINE

        loaded_model = repo.load_model(version.id)
        assert isinstance(loaded_model, RandomForestClassifier)

    def it_raises_when_model_not_found(self, base_path):
        repo = FileSystemModelRepository(
            base_path=base_path,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
        )

        with pytest.raises(Exception):
            repo.load_model("non_existent_model_id")


class DescribeModelVersioningServiceImpl:
    @pytest.fixture
    def repository(self, tmp_path):
        return FileSystemModelRepository(
            base_path=tmp_path / "models",
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
        )

    def it_returns_latest_version(self, repository: ModelRepository):
        repo = repository
        first_version = repo.save_model(RandomForestClassifier(random_state=0), id="first")
        second_version = repo.save_model(RandomForestClassifier(random_state=1), id="second")

        service = ModelVersioningServiceImpl(repo)

        versions = list(service.list_versions(ModelType.RANDOM_FOREST, Technique.BASELINE))
        assert [v.id for v in versions] == [second_version.id, first_version.id]

        latest = service.get_latest_version(ModelType.RANDOM_FOREST, Technique.BASELINE)
        assert latest.id == second_version.id

    def it_raises_when_model_not_found(self, repository: ModelRepository):
        service = ModelVersioningServiceImpl(repository)

        with pytest.raises(Exception):
            service.get_latest_version(ModelType.RANDOM_FOREST, Technique.BASELINE)
