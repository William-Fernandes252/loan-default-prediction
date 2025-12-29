"""Tests for experiments.settings module."""

import os
from pathlib import Path
from unittest.mock import patch

from experiments.settings import (
    ExperimentSettings,
    ExperimentsSettings,
    PathSettings,
    ResourceSettings,
)


class DescribePathSettings:
    """Tests for PathSettings class."""

    class DescribeInit:
        """Tests for initialization."""

        def it_uses_default_project_root(self) -> None:
            """Verify uses default project root when not specified."""
            settings = PathSettings()
            # Should point to the experiments package's parent
            assert settings.project_root.exists()

        def it_accepts_custom_project_root(self, tmp_path: Path) -> None:
            """Verify accepts custom project root."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.project_root == tmp_path

    class DescribeDataDirProperties:
        """Tests for data directory properties."""

        def it_returns_correct_data_dir(self, tmp_path: Path) -> None:
            """Verify data_dir is project_root/data."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.data_dir == tmp_path / "data"

        def it_returns_correct_raw_data_dir(self, tmp_path: Path) -> None:
            """Verify raw_data_dir is data/raw."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.raw_data_dir == tmp_path / "data" / "raw"

        def it_returns_correct_interim_data_dir(self, tmp_path: Path) -> None:
            """Verify interim_data_dir is data/interim."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.interim_data_dir == tmp_path / "data" / "interim"

        def it_returns_correct_processed_data_dir(self, tmp_path: Path) -> None:
            """Verify processed_data_dir is data/processed."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.processed_data_dir == tmp_path / "data" / "processed"

        def it_returns_correct_external_data_dir(self, tmp_path: Path) -> None:
            """Verify external_data_dir is data/external."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.external_data_dir == tmp_path / "data" / "external"

    class DescribeOtherDirProperties:
        """Tests for other directory properties."""

        def it_returns_correct_models_dir(self, tmp_path: Path) -> None:
            """Verify models_dir is project_root/models."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.models_dir == tmp_path / "models"

        def it_returns_correct_results_dir(self, tmp_path: Path) -> None:
            """Verify results_dir is project_root/results."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.results_dir == tmp_path / "results"

        def it_returns_correct_reports_dir(self, tmp_path: Path) -> None:
            """Verify reports_dir is project_root/reports."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.reports_dir == tmp_path / "reports"

        def it_returns_correct_figures_dir(self, tmp_path: Path) -> None:
            """Verify figures_dir is reports/figures."""
            settings = PathSettings(project_root=tmp_path)
            assert settings.figures_dir == tmp_path / "reports" / "figures"


class DescribeExperimentSettings:
    """Tests for ExperimentSettings class."""

    class DescribeDefaults:
        """Tests for default values."""

        def it_has_default_cv_folds(self) -> None:
            """Verify default cv_folds is 5."""
            settings = ExperimentSettings()
            assert settings.cv_folds == 5

        def it_has_default_num_seeds(self) -> None:
            """Verify default num_seeds is 30."""
            settings = ExperimentSettings()
            assert settings.num_seeds == 30

        def it_has_default_cost_grids(self) -> None:
            """Verify default cost_grids has 4 entries."""
            settings = ExperimentSettings()
            assert len(settings.cost_grids) == 4
            assert None in settings.cost_grids
            assert "balanced" in settings.cost_grids

    class DescribeEnvironmentOverride:
        """Tests for environment variable overrides."""

        def it_loads_cv_folds_from_env(self) -> None:
            """Verify cv_folds can be set via environment variable."""
            with patch.dict(os.environ, {"LDP_CV_FOLDS": "10"}):
                settings = ExperimentSettings()
                assert settings.cv_folds == 10

        def it_loads_num_seeds_from_env(self) -> None:
            """Verify num_seeds can be set via environment variable."""
            with patch.dict(os.environ, {"LDP_NUM_SEEDS": "50"}):
                settings = ExperimentSettings()
                assert settings.num_seeds == 50


class DescribeResourceSettings:
    """Tests for ResourceSettings class."""

    class DescribeDefaults:
        """Tests for default values."""

        def it_has_default_safety_factor(self) -> None:
            """Verify default safety_factor is 3.5."""
            settings = ResourceSettings()
            assert settings.safety_factor == 3.5

        def it_has_default_use_gpu(self) -> None:
            """Verify default use_gpu is False."""
            settings = ResourceSettings()
            assert settings.use_gpu is False

    class DescribeEnvironmentOverride:
        """Tests for environment variable overrides."""

        def it_loads_safety_factor_from_env(self) -> None:
            """Verify safety_factor can be set via environment variable."""
            with patch.dict(os.environ, {"LDP_SAFETY_FACTOR": "5.0"}):
                settings = ResourceSettings()
                assert settings.safety_factor == 5.0

        def it_loads_use_gpu_from_env(self) -> None:
            """Verify use_gpu can be set via environment variable."""
            with patch.dict(os.environ, {"LDP_USE_GPU": "true"}):
                settings = ResourceSettings()
                assert settings.use_gpu is True


class DescribeExperimentsSettings:
    """Tests for ExperimentsSettings root class."""

    class DescribeComposition:
        """Tests for settings composition."""

        def it_contains_path_settings(self) -> None:
            """Verify contains PathSettings instance."""
            settings = ExperimentsSettings()
            assert isinstance(settings.paths, PathSettings)

        def it_contains_experiment_settings(self) -> None:
            """Verify contains ExperimentSettings instance."""
            settings = ExperimentsSettings()
            assert isinstance(settings.experiment, ExperimentSettings)

        def it_contains_resource_settings(self) -> None:
            """Verify contains ResourceSettings instance."""
            settings = ExperimentsSettings()
            assert isinstance(settings.resources, ResourceSettings)

    class DescribeNestedAccess:
        """Tests for accessing nested settings."""

        def it_allows_access_to_path_properties(self, tmp_path: Path) -> None:
            """Verify can access path properties through composition."""
            # Create with custom paths
            path_settings = PathSettings(project_root=tmp_path)
            settings = ExperimentsSettings(paths=path_settings)

            assert settings.paths.project_root == tmp_path
            assert settings.paths.data_dir == tmp_path / "data"

        def it_allows_access_to_experiment_properties(self) -> None:
            """Verify can access experiment properties."""
            settings = ExperimentsSettings()

            assert settings.experiment.cv_folds == 5
            assert settings.experiment.num_seeds == 30

        def it_allows_access_to_resource_properties(self) -> None:
            """Verify can access resource properties."""
            settings = ExperimentsSettings()

            assert settings.resources.safety_factor == 3.5
            assert settings.resources.use_gpu is False
