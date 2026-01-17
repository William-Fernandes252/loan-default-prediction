"""Application settings using Pydantic Settings.

This module defines the ExperimentsSettings class which loads configuration
from environment variables and .env files using pydantic-settings.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


class StorageProvider(str, Enum):
    """Storage provider type."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


class StorageSettings(BaseSettings):
    """Storage-related settings.

    Attributes:
        provider (StorageProvider): The storage provider to use.
        base_path (Path): The base path for local storage.
        s3_bucket (str | None): The S3 bucket name (if using S3).
        s3_prefix (str): The S3 prefix for all paths.
        s3_region (str | None): The AWS region for S3.
        s3_endpoint_url (str | None): Custom S3 endpoint URL (for S3-compatible services).
        s3_access_key_id (str | None): AWS access key ID (optional, uses default credentials if not set).
        s3_secret_access_key (str | None): AWS secret access key.
        gcs_bucket (str | None): The GCS bucket name (if using GCS).
        gcs_prefix (str): The GCS prefix for all paths.
        gcs_project (str | None): The GCP project ID.
        gcs_credentials_file (str | None): Path to GCS credentials JSON file.
        cache_dir (Path | None): Local cache directory for cloud storage operations.
    """

    model_config = SettingsConfigDict(
        env_prefix="LDP_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: StorageProvider = Field(
        default=StorageProvider.LOCAL,
        description="Storage provider: 'local', 's3', or 'gcs'",
    )

    base_path: Annotated[Path, Field(default_factory=_get_project_root)]

    # S3-specific settings
    s3_bucket: str | None = Field(
        default=None,
        description="S3 bucket name",
    )
    s3_prefix: str = Field(
        default="",
        description="S3 prefix for all paths",
    )
    s3_region: str | None = Field(
        default=None,
        description="AWS region for S3",
    )
    s3_endpoint_url: str | None = Field(
        default=None,
        description="Custom S3 endpoint URL (for S3-compatible services)",
    )
    s3_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID (optional, uses default credentials if not set)",
    )
    s3_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key",
    )

    # GCS-specific settings
    gcs_bucket: str | None = Field(
        default=None,
        description="GCS bucket name",
    )
    gcs_prefix: str = Field(
        default="",
        description="GCS prefix for all paths",
    )
    gcs_project: str | None = Field(
        default=None,
        description="GCP project ID",
    )
    gcs_credentials_file: str | None = Field(
        default=None,
        description="Path to GCS credentials JSON file",
    )

    # Local cache settings (for cloud storage)
    cache_dir: Path | None = Field(
        default=None,
        description="Local cache directory for cloud storage operations",
    )


class PathSettings(BaseSettings):
    """Path-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="LDP_",
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
        env_prefix="LDP_",
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
        env_prefix="LDP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    safety_factor: float = Field(
        default=3.5,
        description="Multiplier for peak memory usage in job calculation",
    )
    use_gpu: bool = Field(default=False, description="Whether to use GPU acceleration")


class LdpSettings(BaseSettings):
    """Root settings class that composes all settings groups."""

    model_config = SettingsConfigDict(
        env_prefix="LDP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
