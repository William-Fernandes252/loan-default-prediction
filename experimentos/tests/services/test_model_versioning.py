"""Tests for experiments.services.model_versioning module."""

from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

from experiments.core.modeling.types import ModelType, Technique
from experiments.services.model_versioning import ModelVersioningServiceFactory
from experiments.services.models import ModelVersioningServiceImpl


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    return models


@pytest.fixture
def factory(models_dir: Path) -> ModelVersioningServiceFactory:
    """Create a ModelVersioningServiceFactory."""
    return ModelVersioningServiceFactory(models_dir)


class DescribeModelVersioningServiceFactory:
    """Tests for ModelVersioningServiceFactory class."""

    class DescribeInit:
        """Tests for __init__ method."""

        def it_stores_models_dir(self, models_dir: Path) -> None:
            """Verify stores the models directory."""
            factory = ModelVersioningServiceFactory(models_dir)

            # Access private attribute for testing
            assert factory._models_dir == models_dir

    class DescribeGetModelVersioningService:
        """Tests for get_model_versioning_service method."""

        def it_returns_model_versioning_service_impl(
            self,
            factory: ModelVersioningServiceFactory,
        ) -> None:
            """Verify returns a ModelVersioningServiceImpl instance."""
            service = factory.get_model_versioning_service(
                dataset_id="taiwan_credit",
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
            )

            assert isinstance(service, ModelVersioningServiceImpl)

        def it_returns_new_service_for_different_configs(
            self,
            factory: ModelVersioningServiceFactory,
        ) -> None:
            """Verify returns different services for different configurations."""
            service1 = factory.get_model_versioning_service(
                dataset_id="taiwan_credit",
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
            )
            service2 = factory.get_model_versioning_service(
                dataset_id="taiwan_credit",
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.SMOTE,
            )

            assert service1 is not service2

        def it_creates_functional_service(
            self,
            factory: ModelVersioningServiceFactory,
        ) -> None:
            """Verify created service can save and load models."""
            service = factory.get_model_versioning_service(
                dataset_id="taiwan_credit",
                model_type=ModelType.RANDOM_FOREST,
                technique=Technique.BASELINE,
            )

            # Create a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)

            # Save through the service's repository
            version = service.repository.save_model(model, id="test_model")

            assert version.id == "test_model"
            assert version.type is ModelType.RANDOM_FOREST
            assert version.technique is Technique.BASELINE

        def it_works_with_different_model_types(
            self,
            factory: ModelVersioningServiceFactory,
        ) -> None:
            """Verify works with different model types."""
            for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.SVM]:
                service = factory.get_model_versioning_service(
                    dataset_id="test_dataset",
                    model_type=model_type,
                    technique=Technique.BASELINE,
                )

                assert isinstance(service, ModelVersioningServiceImpl)

        def it_works_with_different_techniques(
            self,
            factory: ModelVersioningServiceFactory,
        ) -> None:
            """Verify works with different techniques."""
            for technique in [
                Technique.BASELINE,
                Technique.SMOTE,
                Technique.SMOTE_TOMEK,
                Technique.RANDOM_UNDER_SAMPLING,
                Technique.META_COST,
            ]:
                service = factory.get_model_versioning_service(
                    dataset_id="test_dataset",
                    model_type=ModelType.RANDOM_FOREST,
                    technique=technique,
                )

                assert isinstance(service, ModelVersioningServiceImpl)
