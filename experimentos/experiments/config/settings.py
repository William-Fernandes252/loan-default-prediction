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
    """Storage-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="LDP_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Annotated[
        StorageProvider,
        Field(
            default=StorageProvider.LOCAL,
        ),
    ]
    """Storage provider to use."""

    base_path: Annotated[Path, Field(default_factory=_get_project_root)]
    """Base path for local storage."""

    # S3-specific settings
    s3_bucket: Annotated[
        str | None,
        Field(
            default=None,
        ),
    ] = None
    """S3 bucket name"""

    s3_prefix: Annotated[
        str,
        Field(
            default="",
            description="S3 prefix for all paths",
        ),
    ] = ""
    """S3 prefix for all paths"""

    s3_region: Annotated[
        str | None,
        Field(
            default=None,
            description="AWS region for S3",
        ),
    ] = None
    """AWS region for S3"""

    s3_endpoint_url: Annotated[
        str | None,
        Field(
            default=None,
            description="Custom S3 endpoint URL (for S3-compatible services)",
        ),
    ] = None
    """Custom S3 endpoint URL (for S3-compatible services)"""

    s3_access_key_id: Annotated[
        str | None,
        Field(
            default=None,
            description="AWS access key ID (optional, uses default credentials if not set)",
        ),
    ] = None
    """AWS access key ID (optional, uses default credentials if not set)"""

    s3_secret_access_key: Annotated[
        str | None,
        Field(
            default=None,
            description="AWS secret access key",
        ),
    ] = None
    """AWS secret access key"""

    # GCS-specific settings
    gcs_bucket: Annotated[
        str | None,
        Field(
            default=None,
            description="GCS bucket name",
        ),
    ] = None
    """GCS bucket name"""

    gcs_prefix: Annotated[
        str,
        Field(
            default="",
            description="GCS prefix for all paths",
        ),
    ] = ""
    """GCS prefix for all paths"""

    gcs_project: Annotated[
        str | None,
        Field(
            default=None,
            description="GCP project ID",
        ),
    ] = None
    """GCP project ID"""

    gcs_credentials_file: Annotated[
        str | None,
        Field(
            default=None,
            description="Path to GCS credentials JSON file",
        ),
    ] = None
    """Path to GCS credentials JSON file"""

    cache_dir: Annotated[
        Path | None,
        Field(
            default=None,
            description="Local cache directory for cloud storage operations",
        ),
    ] = None
    """Local cache directory for cloud storage operations"""


class ExperimentSettings(BaseSettings):
    """Experiment-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="LDP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sampler_k_neighbors: Annotated[int, Field(default=3)]
    """Number of neighbors for over-sampling."""

    cv_folds: Annotated[int, Field(default=5)]
    """Number of cross-validation folds."""

    num_seeds: Annotated[int, Field(default=30)]
    """Number of random seeds to use for experiments."""

    cost_grids: Annotated[
        list[Any],
        Field(
            default_factory=lambda: [
                None,  # Default weight (1:1)
                "balanced",  # Inverse frequency (sklearn default)
                {0: 1, 1: 5},  # 5x weight for minority class error
                {0: 1, 1: 10},  # 10x weight for minority class error
            ]
        ),
    ]
    """Cost grids for optimization."""


class ResourceSettings(BaseSettings):
    """Resource-related settings."""

    model_config = SettingsConfigDict(
        env_prefix="LDP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    safety_factor: Annotated[float, Field(default=3.5)]
    """Multiplier for peak memory usage in job calculation."""

    use_gpu: Annotated[bool, Field(default=False)]
    """Whether to use GPU for model training if available."""

    n_jobs: Annotated[int, Field(default=1)]
    """Default number of parallel jobs for processing."""

    models_n_jobs: Annotated[int, Field(default=1)]
    """Number of parallel jobs for model training."""

    sequential: Annotated[bool, Field(default=False)]
    """Whether to run training pipelines sequentially instead of in parallel."""


class LdpSettings(BaseSettings):
    """Root settings class that composes all settings groups."""

    model_config = SettingsConfigDict(
        env_prefix="LDP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    debug: Annotated[bool, Field(default=True)] = True
    """Debug mode flag."""

    sentry_dns: Annotated[
        str | None,
        Field(default=None),
    ] = None
    """Sentry DSN for error tracking (if not set, Sentry is disabled)."""

    locale: Annotated[
        str,
        Field(
            default="pt_BR",
            description="Default locale for analysis artifact generation (en_US or pt_BR)",
        ),
    ]
    """Default locale for generating analysis artifacts."""
