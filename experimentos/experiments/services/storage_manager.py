"""Storage management service for the experiments application.

This module provides a StorageManager class that encapsulates all storage
operations and URI resolution, replacing direct filesystem access with
a storage abstraction layer that supports local and cloud backends.
"""

from contextlib import contextmanager
from datetime import datetime
import gc
import os
import tempfile
from typing import TYPE_CHECKING, Any, Generator

import joblib
from loguru import logger
import polars as pl

from experiments.services.storage import StorageService
from experiments.settings import PathSettings

if TYPE_CHECKING:
    from experiments.core.data import Dataset


class StorageManager:
    """Manages all storage operations for the experiments application.

    This class provides a unified interface for all file operations,
    supporting both local filesystem and cloud storage backends through
    the StorageService abstraction.

    All methods return URIs (strings) instead of Path objects, following
    the format: file://, s3://, or gs:// depending on the storage backend.

    Attributes:
        settings: The path settings configuration.
        storage: The storage service for file operations.
    """

    def __init__(
        self,
        settings: PathSettings,
        storage: StorageService,
    ) -> None:
        """Initialize the StorageManager.

        Args:
            settings: Path settings configuration (used for base paths).
            storage: Storage service implementation.
        """
        self._settings = settings
        self._storage = storage

    @property
    def storage(self) -> StorageService:
        """Get the underlying storage service."""
        return self._storage

    # --- URI Construction Helpers ---

    def _to_uri(self, *parts: str) -> str:
        """Construct a URI from path parts."""
        path = os.path.join(*parts)
        return StorageService.to_uri(path)

    # --- Raw/Interim Data URIs ---

    def get_raw_data_uri(self, dataset_id: str) -> str:
        """Get the URI to the raw CSV data file for a dataset.

        Args:
            dataset_id: The dataset identifier (e.g., 'taiwan_credit').

        Returns:
            URI to the raw data CSV file.
        """
        return self._to_uri(str(self._settings.raw_data_dir), f"{dataset_id}.csv")

    def get_interim_data_uri(self, dataset_id: str) -> str:
        """Get the URI to the interim parquet data file for a dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            URI to the interim parquet file.
        """
        return self._to_uri(str(self._settings.interim_data_dir), f"{dataset_id}.parquet")

    # --- Feature URIs ---

    def get_feature_uris(self, dataset_id: str) -> dict[str, str]:
        """Get URIs to feature X and target y parquet files.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            Dictionary with 'X' and 'y' keys mapping to their URIs.
        """
        base = str(self._settings.processed_data_dir)
        return {
            "X": self._to_uri(base, f"{dataset_id}_X.parquet"),
            "y": self._to_uri(base, f"{dataset_id}_y.parquet"),
        }

    # --- Checkpoint URIs ---

    def get_checkpoint_uri(
        self,
        dataset_id: str,
        model_id: str,
        technique_id: str,
        seed: int,
    ) -> str:
        """Get the checkpoint URI for a specific experiment task.

        Args:
            dataset_id: The dataset identifier.
            model_id: The model type identifier.
            technique_id: The technique identifier.
            seed: The random seed.

        Returns:
            URI to the checkpoint parquet file.
        """
        ckpt_dir = os.path.join(
            str(self._settings.results_dir),
            dataset_id,
            "checkpoints",
        )
        # Ensure directory exists
        self._storage.makedirs(self._to_uri(ckpt_dir))
        return self._to_uri(ckpt_dir, f"{model_id}_{technique_id}_seed{seed}.parquet")

    def get_checkpoints_dir_uri(self, dataset_id: str) -> str:
        """Get the checkpoints directory URI for a dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            URI to the checkpoints directory.
        """
        return self._to_uri(
            str(self._settings.results_dir),
            dataset_id,
            "checkpoints",
        )

    # --- Consolidated Results URIs ---

    def get_consolidated_results_uri(self, dataset_id: str) -> str:
        """Get a new consolidated results URI with timestamp.

        Creates a timestamped filename for consolidated results.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            URI to the new consolidated results file.

        Example:
            `file:///path/to/results/taiwan_credit/20251225_153045.parquet`
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(str(self._settings.results_dir), dataset_id)
        self._storage.makedirs(self._to_uri(results_dir))
        return self._to_uri(results_dir, f"{ts}.parquet")

    def get_latest_consolidated_results_uri(self, dataset_id: str) -> str | None:
        """Get the most recent consolidated results file for a dataset.

        This is useful for analysis code that wants to read the latest
        consolidated artifact without knowing the exact timestamped filename.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            URI to the most recent results file, or None if none exist.
        """
        results_dir_uri = self._to_uri(str(self._settings.results_dir), dataset_id)

        # List files matching the timestamp pattern
        pattern = "[0-9]" * 8 + "_" + "[0-9]" * 6 + ".parquet"
        files = self._storage.list_files(results_dir_uri, pattern)

        if not files:
            return None

        # For cloud storage, we can't easily get mtime, so sort by filename
        # (which is a timestamp, so alphabetical order = chronological order)
        return sorted(files)[-1]

    # --- Directory URIs ---

    def get_dataset_results_dir_uri(self, dataset_id: str, *, create: bool = False) -> str:
        """Get the results directory URI for a dataset.

        Args:
            dataset_id: The dataset identifier.
            create: Whether to create the directory if it doesn't exist.

        Returns:
            URI to the dataset's results directory.
        """
        uri = self._to_uri(str(self._settings.results_dir), dataset_id)
        if create:
            self._storage.makedirs(uri)
        return uri

    def get_dataset_figures_dir_uri(self, dataset_id: str, *, create: bool = False) -> str:
        """Get the figures directory URI for a dataset.

        Args:
            dataset_id: The dataset identifier.
            create: Whether to create the directory if it doesn't exist.

        Returns:
            URI to the dataset's figures directory.
        """
        uri = self._to_uri(str(self._settings.figures_dir), dataset_id)
        if create:
            self._storage.makedirs(uri)
        return uri

    def get_models_dir_uri(self) -> str:
        """Get the root models directory URI."""
        return self._to_uri(str(self._settings.models_dir))

    # --- Data Operations ---

    def read_raw_data(self, dataset_id: str, **kwargs: Any) -> pl.DataFrame:
        """Read raw CSV data for a dataset.

        Args:
            dataset_id: The dataset identifier.
            **kwargs: Additional arguments passed to read_csv.

        Returns:
            DataFrame containing the raw data.
        """
        uri = self.get_raw_data_uri(dataset_id)
        return self._storage.read_csv(uri, **kwargs)

    def write_interim_data(self, df: pl.DataFrame, dataset_id: str) -> str:
        """Write interim data to parquet.

        Args:
            df: The DataFrame to write.
            dataset_id: The dataset identifier.

        Returns:
            URI where the data was written.
        """
        uri = self.get_interim_data_uri(dataset_id)
        self._storage.write_parquet(df, uri)
        return uri

    def read_features(self, dataset_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Read feature X and target y DataFrames.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            Tuple of (X, y) DataFrames.
        """
        uris = self.get_feature_uris(dataset_id)
        X = self._storage.read_parquet(uris["X"])
        y = self._storage.read_parquet(uris["y"])
        return X, y

    def write_features(
        self,
        X: pl.DataFrame,
        y: pl.DataFrame,
        dataset_id: str,
    ) -> dict[str, str]:
        """Write feature X and target y DataFrames.

        Args:
            X: Features DataFrame.
            y: Target DataFrame.
            dataset_id: The dataset identifier.

        Returns:
            Dictionary with 'X' and 'y' keys mapping to their URIs.
        """
        uris = self.get_feature_uris(dataset_id)
        self._storage.write_parquet(X, uris["X"])
        self._storage.write_parquet(y, uris["y"])
        return uris

    def features_exist(self, dataset_id: str) -> bool:
        """Check if feature artifacts exist for a dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            True if both X and y feature files exist.
        """
        uris = self.get_feature_uris(dataset_id)
        return self._storage.exists(uris["X"]) and self._storage.exists(uris["y"])

    def artifacts_exist(self, dataset: "Dataset") -> bool:
        """Check if feature artifacts exist for a dataset (DataProvider protocol).

        Args:
            dataset: The dataset to check.

        Returns:
            True if both X and y feature files exist.
        """
        exists = self.features_exist(dataset.id)
        if not exists:
            logger.warning(f"Artifacts not found for {dataset.display_name}.")
        return exists

    # --- Checkpoint Operations ---

    def write_checkpoint(self, df: pl.DataFrame, uri: str) -> None:
        """Write a checkpoint DataFrame.

        Args:
            df: The checkpoint DataFrame to write.
            uri: The URI where the checkpoint will be written.
        """
        self._storage.write_parquet(df, uri)

    def read_checkpoint(self, uri: str) -> pl.DataFrame:
        """Read a checkpoint DataFrame.

        Args:
            uri: The URI of the checkpoint file.

        Returns:
            The checkpoint DataFrame.
        """
        return self._storage.read_parquet(uri)

    def checkpoint_exists(self, uri: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            uri: The URI of the checkpoint.

        Returns:
            True if the checkpoint exists.
        """
        return self._storage.exists(uri)

    def delete_checkpoint(self, uri: str) -> None:
        """Delete a checkpoint.

        Args:
            uri: The URI of the checkpoint to delete.
        """
        self._storage.delete(uri)

    def list_checkpoints(self, dataset_id: str) -> list[str]:
        """List all checkpoints for a dataset.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            List of checkpoint URIs.
        """
        ckpt_dir_uri = self.get_checkpoints_dir_uri(dataset_id)
        return self._storage.list_files(ckpt_dir_uri, "*.parquet")

    # --- Results Consolidation ---

    def consolidate_checkpoints(self, dataset_id: str) -> str | None:
        """Consolidate all checkpoints for a dataset into a single results file.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            URI to the consolidated results file, or None if no checkpoints.
        """
        checkpoint_uris = self.list_checkpoints(dataset_id)

        if not checkpoint_uris:
            logger.warning(f"No checkpoints found for {dataset_id}")
            return None

        logger.info(f"Consolidating {len(checkpoint_uris)} checkpoints for {dataset_id}...")

        frames = []
        for uri in checkpoint_uris:
            try:
                frames.append(self._storage.read_parquet(uri))
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {uri}: {e}")

        if not frames:
            logger.warning("No valid checkpoint frames could be loaded.")
            return None

        df_final = pl.concat(frames)
        output_uri = self.get_consolidated_results_uri(dataset_id)
        self._storage.write_parquet(df_final, output_uri)

        logger.success(f"Saved consolidated results to {output_uri}")
        return output_uri

    # --- Memory-Mapped Feature Context ---

    @contextmanager
    def feature_context(
        self,
        dataset: "Dataset",
    ) -> Generator[tuple[str, str], None, None]:
        """Context manager that prepares data for parallel access with lifecycle safety.

        This context manager ensures proper cleanup of memory-mapped data:

        1. Loads Parquet files into memory via storage layer.
        2. Dumps them to a temporary memory-mapped file (joblib format).
        3. Yields the paths to these memory maps.
        4. Automatically cleans up temporary files on context exit.

        **IMPORTANT - Memory Management:**
        - Consumers MUST NOT copy data; use array slicing to create views.
        - Example: `X_train = X_mmap[indices]` creates a view, NOT a copy.
        - Views remain valid only while this context is active.
        - The context manager ensures mmap files are unlocked and deleted on exit.
        - Do NOT store references to mmap arrays beyond this context's scope.

        For cloud storage, files are first downloaded to local cache.

        Yields:
            tuple[str, str]: Paths to (X_mmap, y_mmap).

        Raises:
            FileNotFoundError: If feature artifacts don't exist.
        """
        uris = self.get_feature_uris(dataset.id)

        if not self.features_exist(dataset.id):
            raise FileNotFoundError(f"Data missing for {dataset.display_name}")

        logger.info(f"Loading data for {dataset.display_name}...")

        # Load original data via storage layer
        # Convert Polars to Pandas/NumPy for joblib memory-mapping
        X_df = self._storage.read_parquet(uris["X"]).to_pandas()
        y_df = self._storage.read_parquet(uris["y"]).to_pandas().iloc[:, 0]

        # Create a temporary directory for memory mapping
        with tempfile.TemporaryDirectory() as temp_dir:
            X_mmap_path = os.path.join(temp_dir, "X.mmap")
            y_mmap_path = os.path.join(temp_dir, "y.mmap")

            # Dump data using joblib (optimized for numpy persistence)
            joblib.dump(X_df.to_numpy(), X_mmap_path)
            joblib.dump(y_df.to_numpy(), y_mmap_path)

            # Quick verification load to ensure integrity
            _ = joblib.load(X_mmap_path, mmap_mode="r")

            # Aggressively free the original RAM before yielding control
            del X_df, y_df
            gc.collect()

            logger.info(f"Data memory-mapped for {dataset.display_name}")

            try:
                yield X_mmap_path, y_mmap_path
            finally:
                # Context exit: temp_dir cleanup handles file deletion.
                # GC helps ensure file handles are released for proper unlink.
                gc.collect()

    def get_dataset_size_gb(self, dataset: "Dataset") -> float:
        """Estimate the size of the raw dataset in GB for job calculation.

        This method implements the DataProvider protocol.

        Args:
            dataset: The dataset to check.

        Returns:
            Estimated size in GB.
        """
        uri = self.get_raw_data_uri(dataset.id)
        try:
            return self._storage.get_size_bytes(uri) / (1024**3)
        except Exception:
            return 1.0  # Default fallback

    # --- Model Operations ---

    def save_model(self, model: Any, uri: str) -> None:
        """Save a model using joblib.

        Args:
            model: The model to save.
            uri: The URI where the model will be saved.
        """
        self._storage.write_joblib(model, uri)

    def load_model(self, uri: str) -> Any:
        """Load a model using joblib.

        Args:
            uri: The URI of the model file.

        Returns:
            The loaded model.
        """
        return self._storage.read_joblib(uri)

    def save_model_metadata(self, metadata: dict[str, Any], uri: str) -> None:
        """Save model metadata as JSON.

        Args:
            metadata: The metadata dictionary.
            uri: The URI where the metadata will be saved.
        """
        self._storage.write_json(metadata, uri)

    def load_model_metadata(self, uri: str) -> dict[str, Any]:
        """Load model metadata from JSON.

        Args:
            uri: The URI of the metadata file.

        Returns:
            The metadata dictionary.
        """
        return self._storage.read_json(uri)
