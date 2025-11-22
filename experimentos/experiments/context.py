"""Context management for experiments.

This module defines the Context class, which encapsulates configuration,
logging, and environment helpers. It serves as the primary interface
for passing state between the CLI and core logic.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from loguru import logger
import psutil

from experiments import config


@dataclass
class AppConfig:
    """Application configuration parameters."""

    # Paths
    raw_data_dir: Path = config.RAW_DATA_DIR
    interim_data_dir: Path = config.INTERIM_DATA_DIR
    processed_data_dir: Path = config.PROCESSED_DATA_DIR
    results_dir: Path = config.RESULTS_DIR
    models_dir: Path = config.MODELS_DIR

    # Experiment Settings
    cost_grids: list[Any] = field(default_factory=lambda: config.COST_GRIDS)
    cv_folds: int = config.CV_FOLDS
    num_seeds: int = config.NUM_SEEDS

    # Environment
    max_threads: str = "1"
    use_gpu: bool = False


class Context:
    """
    The runtime context for the application.

    Acts as a facade for configuration, logging, and system interactions.
    """

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()
        self.logger = logger
        self._configure_environment()

    def _configure_environment(self) -> None:
        """Sets up environment variables for numerical libraries."""
        # Limit thread usage per worker to prevent thread explosion in parallel execution
        os.environ["OMP_NUM_THREADS"] = self.cfg.max_threads
        os.environ["MKL_NUM_THREADS"] = self.cfg.max_threads
        os.environ["OPENBLAS_NUM_THREADS"] = self.cfg.max_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = self.cfg.max_threads
        os.environ["NUMEXPR_NUM_THREADS"] = self.cfg.max_threads

    # --- Path Helpers ---

    def get_raw_data_path(self, dataset_value: str) -> Path:
        return self.cfg.raw_data_dir / f"{dataset_value}.csv"

    def get_interim_data_path(self, dataset_value: str) -> Path:
        return self.cfg.interim_data_dir / f"{dataset_value}.parquet"

    def get_feature_paths(self, dataset_value: str) -> dict[str, Path]:
        base = self.cfg.processed_data_dir
        return {
            "X": base / f"{dataset_value}_X.parquet",
            "y": base / f"{dataset_value}_y.parquet",
        }

    def get_checkpoint_path(
        self, dataset_value: str, model: str, technique: str, seed: int
    ) -> Path:
        ckpt_dir = self.cfg.results_dir / "checkpoints" / dataset_value
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"{model}_{technique}_seed{seed}.parquet"

    def get_consolidated_results_path(self, dataset_value: str) -> Path:
        return self.cfg.results_dir / f"{dataset_value}_results.parquet"

    # --- Resource Helpers ---

    def compute_safe_jobs(self, dataset_size_gb: float, safety_factor: float = 3.5) -> int:
        """
        Calculates safe number of parallel jobs based on available RAM.

        Args:
            dataset_size_gb: Size of the dataset in GB.
            safety_factor: Multiplier for peak memory usage.
        """
        # Get available RAM in GB
        available_ram_gb = psutil.virtual_memory().available / (1024**3)

        # Estimate peak memory required per worker
        # Only ~70% of data is loaded for training in splits
        train_size_gb = dataset_size_gb * 0.70
        peak_memory_per_worker = train_size_gb * safety_factor

        # Avoid division by zero for tiny datasets
        if peak_memory_per_worker < 0.1:
            peak_memory_per_worker = 0.1

        # Calculate jobs
        safe_jobs = int(available_ram_gb // peak_memory_per_worker)

        # Ensure at least 1 job and don't exceed CPU count
        cpu_count = psutil.cpu_count(logical=False) or 1
        return max(1, min(safe_jobs, cpu_count))


# Global instance for default usage if needed,
# though it's better to pass it around.
global_context = Context()
