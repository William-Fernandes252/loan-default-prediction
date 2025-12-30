"""Tests for experiments.containers module."""

from pathlib import Path

import pytest

from experiments.containers import Container
from experiments.core.modeling.factories import DefaultEstimatorFactory
from experiments.services.model_versioning import ModelVersioningServiceFactory
from experiments.services.path_manager import PathManager
from experiments.services.resource_calculator import ResourceCalculator
from experiments.settings import ExperimentsSettings, PathSettings


@pytest.fixture
def container(tmp_path: Path) -> Container:
    """Create a Container with temporary paths for testing."""
    container = Container()

    # Override settings with test paths
    test_path_settings = PathSettings(project_root=tmp_path)
    test_settings = ExperimentsSettings(paths=test_path_settings)
    container.settings.override(test_settings)

    return container


class DescribeContainer:
    """Tests for Container class."""

    class DescribeSettingsProvider:
        """Tests for settings provider."""

        def it_provides_experiments_settings(self, container: Container) -> None:
            """Verify provides ExperimentsSettings instance."""
            settings = container.settings()

            assert isinstance(settings, ExperimentsSettings)

        def it_provides_singleton_settings(self, container: Container) -> None:
            """Verify settings is a singleton."""
            settings1 = container.settings()
            settings2 = container.settings()

            assert settings1 is settings2

    class DescribePathManagerProvider:
        """Tests for path_manager provider."""

        def it_provides_path_manager(self, container: Container) -> None:
            """Verify provides PathManager instance."""
            path_manager = container.path_manager()

            assert isinstance(path_manager, PathManager)

        def it_provides_singleton_path_manager(self, container: Container) -> None:
            """Verify path_manager is a singleton."""
            pm1 = container.path_manager()
            pm2 = container.path_manager()

            assert pm1 is pm2

        def it_uses_settings_paths(self, container: Container, tmp_path: Path) -> None:
            """Verify path_manager uses settings.paths."""
            path_manager = container.path_manager()

            # The path_manager should use the overridden tmp_path
            assert path_manager.models_dir == tmp_path / "models"

    class DescribeResourceCalculatorProvider:
        """Tests for resource_calculator provider."""

        def it_provides_resource_calculator(self, container: Container) -> None:
            """Verify provides ResourceCalculator instance."""
            calc = container.resource_calculator()

            assert isinstance(calc, ResourceCalculator)

        def it_provides_singleton_resource_calculator(self, container: Container) -> None:
            """Verify resource_calculator is a singleton."""
            calc1 = container.resource_calculator()
            calc2 = container.resource_calculator()

            assert calc1 is calc2

        def it_uses_settings_safety_factor(self, container: Container) -> None:
            """Verify uses safety_factor from settings."""
            settings = container.settings()
            calc = container.resource_calculator()

            assert calc.safety_factor == settings.resources.safety_factor

    class DescribeModelVersioningFactoryProvider:
        """Tests for model_versioning_factory provider."""

        def it_provides_model_versioning_factory(self, container: Container) -> None:
            """Verify provides ModelVersioningServiceFactory instance."""
            factory = container.model_versioning_factory()

            assert isinstance(factory, ModelVersioningServiceFactory)

        def it_provides_singleton_factory(self, container: Container) -> None:
            """Verify model_versioning_factory is a singleton."""
            f1 = container.model_versioning_factory()
            f2 = container.model_versioning_factory()

            assert f1 is f2

    class DescribeStorageManagerProvider:
        """Tests for storage_manager provider."""

        def it_provides_storage_manager(self, container: Container) -> None:
            """Verify provides StorageManager instance."""
            sm = container.storage_manager()

            from experiments.services.storage_manager import StorageManager

            assert isinstance(sm, StorageManager)

        def it_provides_singleton_storage_manager(self, container: Container) -> None:
            """Verify storage_manager is a singleton."""
            sm1 = container.storage_manager()
            sm2 = container.storage_manager()

            # Singleton provider returns same instance
            assert sm1 is sm2

    class DescribeOverrides:
        """Tests for provider overrides."""

        def it_allows_settings_override(self, tmp_path: Path) -> None:
            """Verify settings can be overridden."""
            container = Container()

            custom_path = tmp_path / "custom"
            custom_settings = ExperimentsSettings(paths=PathSettings(project_root=custom_path))
            container.settings.override(custom_settings)

            settings = container.settings()
            assert settings.paths.project_root == custom_path

        def it_allows_path_manager_override(self, tmp_path: Path) -> None:
            """Verify path_manager can be overridden."""
            container = Container()

            custom_pm = PathManager(PathSettings(project_root=tmp_path))
            container.path_manager.override(custom_pm)

            pm = container.path_manager()
            assert pm is custom_pm

    class DescribeReset:
        """Tests for container reset functionality."""

        def it_can_reset_singleton_provider_overrides(self) -> None:
            """Verify singleton provider overrides can be reset."""
            container = Container()

            from experiments.services.storage import LocalStorageService
            from experiments.services.storage_manager import StorageManager

            # Create a mock storage manager
            storage = LocalStorageService()
            mock_sm = StorageManager(storage, PathSettings())

            # Override storage_manager (singleton provider)
            container.storage_manager.override(mock_sm)

            # Verify override is in effect
            assert container.storage_manager() is mock_sm

            # Reset the override using reset_override()
            container.storage_manager.reset_override()

            # After reset, should create new instance (not the mock)
            new_sm = container.storage_manager()
            assert new_sm is not mock_sm
            assert isinstance(new_sm, StorageManager)

    class DescribeEstimatorFactoryProvider:
        """Tests for estimator_factory provider."""

        def it_provides_estimator_factory(self, container: Container) -> None:
            """Verify provides DefaultEstimatorFactory instance."""
            factory = container.estimator_factory()

            assert isinstance(factory, DefaultEstimatorFactory)

        def it_provides_singleton_factory(self, container: Container) -> None:
            """Verify estimator_factory is a singleton."""
            f1 = container.estimator_factory()
            f2 = container.estimator_factory()

            assert f1 is f2

        def it_uses_settings_use_gpu(self, container: Container) -> None:
            """Verify uses use_gpu from settings."""
            settings = container.settings()
            factory = container.estimator_factory()

            # Factory should be configured with the use_gpu setting
            assert factory._use_gpu == settings.resources.use_gpu
