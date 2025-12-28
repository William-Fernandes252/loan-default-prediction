"""Application settings using Pydantic Settings.

This module defines the ExperimentsSettings class which loads configuration
from environment variables and .env files using pydantic-settings.
"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[1]


class PathSettings(BaseSettings):
    """Path-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_get_project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def external_data_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


class ExperimentSettings(BaseSettings):
    """Experiment-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    num_seeds: int = Field(default=30, description="Number of experiment repetitions")
    cost_grids: list[Any] = Field(
        default_factory=lambda: [
            None,  # Default weight (1:1)
            "balanced",  # Inverse frequency (sklearn default)
            {0: 1, 1: 5},  # 5x weight for minority class error
            {0: 1, 1: 10},  # 10x weight for minority class error
        ],
        description="Cost grids for optimization",
    )


class ResourceSettings(BaseSettings):
    """Resource-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    safety_factor: float = Field(
        default=3.5,
        description="Multiplier for peak memory usage in job calculation",
    )
    use_gpu: bool = Field(default=False, description="Whether to use GPU acceleration")


class ExperimentsSettings(BaseSettings):
    """Root settings class that composes all settings groups."""

    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)
