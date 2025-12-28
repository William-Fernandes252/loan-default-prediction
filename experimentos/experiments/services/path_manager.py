"""Path management service for the experiments application.

This module provides a PathManager class that encapsulates all path resolution
logic, implementing the various path provider protocols used by the pipelines.
"""

from datetime import datetime
from pathlib import Path

from experiments.settings import PathSettings


class PathManager:
    """Manages all path resolution for the experiments application.

    This class implements the following protocols:
    - RawDataPathProvider
    - InterimDataPathProvider
    - CheckpointPathProvider
    - ConsolidatedResultsPathProvider
    - ResultsPathProvider (via get_latest_consolidated_results_path)

    Attributes:
        settings: The path settings configuration.
    """

    def __init__(self, settings: PathSettings) -> None:
        """Initialize the PathManager.

        Args:
            settings: Path settings configuration.
        """
        self._settings = settings

    # --- Raw/Interim Data Paths ---

    def get_raw_data_path(self, dataset_id: str) -> Path:
        """Get the path to the raw CSV data file for a dataset.

        Args:
            dataset_id: The dataset identifier (e.g., 'taiwan_credit').

        Returns:
            Path to the raw data CSV file.
        """
        return self._settings.raw_data_dir / f"{dataset_id}.csv"

    def get_interim_data_path(self, dataset_id: str) -> Path:
        """Get the path to the interim parquet data file for a dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            Path to the interim parquet file.
        """
        return self._settings.interim_data_dir / f"{dataset_id}.parquet"

    # --- Feature Paths ---

    def get_feature_paths(self, dataset_id: str) -> dict[str, Path]:
        """Get paths to feature X and target y parquet files.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            Dictionary with 'X' and 'y' keys mapping to their paths.
        """
        base = self._settings.processed_data_dir
        return {
            "X": base / f"{dataset_id}_X.parquet",
            "y": base / f"{dataset_id}_y.parquet",
        }

    # --- Checkpoint Paths ---

    def get_checkpoint_path(
        self,
        dataset_id: str,
        model_id: str,
        technique_id: str,
        seed: int,
    ) -> Path:
        """Get the checkpoint path for a specific experiment task.

        Args:
            dataset_id: The dataset identifier.
            model_id: The model type identifier.
            technique_id: The technique identifier.
            seed: The random seed.

        Returns:
            Path to the checkpoint parquet file.
        """
        dataset_dir = self.get_dataset_results_dir(dataset_id, create=True)
        ckpt_dir = dataset_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"{model_id}_{technique_id}_seed{seed}.parquet"

    # --- Consolidated Results Paths ---

    def get_consolidated_results_path(self, dataset_id: str) -> Path:
        """Get a new consolidated results path with timestamp.

        Creates a timestamped filename for consolidated results.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            Path to the new consolidated results file.

        Example:
            `results/taiwan_credit/20251225_153045.parquet`
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.get_dataset_results_dir(dataset_id, create=True)
        return dataset_dir / f"{ts}.parquet"

    def get_latest_consolidated_results_path(self, dataset_id: str) -> Path | None:
        """Get the most recent consolidated results file for a dataset.

        This is useful for analysis code that wants to read the latest
        consolidated artifact without knowing the exact timestamped filename.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            Path to the most recent results file, or None if none exist.
        """
        dataset_dir = self.get_dataset_results_dir(dataset_id)
        if not dataset_dir.exists():
            return None

        pattern = f"{'[0-9]' * 8}_{'[0-9]' * 6}.parquet"
        files = list(dataset_dir.glob(pattern))
        if not files:
            return None
        # Choose the most recently modified file
        return max(files, key=lambda p: p.stat().st_mtime)

    # --- Directory Paths ---

    def get_dataset_results_dir(self, dataset_id: str, *, create: bool = False) -> Path:
        """Get the results directory for a dataset.

        Args:
            dataset_id: The dataset identifier.
            create: Whether to create the directory if it doesn't exist.

        Returns:
            Path to the dataset's results directory.
        """
        dataset_dir = self._settings.results_dir / dataset_id
        if create:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def get_dataset_figures_dir(self, dataset_id: str, *, create: bool = False) -> Path:
        """Get the figures directory for a dataset.

        Args:
            dataset_id: The dataset identifier.
            create: Whether to create the directory if it doesn't exist.

        Returns:
            Path to the dataset's figures directory.
        """
        dataset_dir = self._settings.figures_dir / dataset_id
        if create:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    @property
    def models_dir(self) -> Path:
        """Get the root models directory."""
        return self._settings.models_dir
