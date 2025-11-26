"""Service for managing experiment data, memory mapping, and result artifacts."""

from contextlib import contextmanager
import gc
import os
import tempfile
from typing import Generator, cast

import joblib
import pandas as pd
import polars as pl

from experiments.context import Context
from experiments.core.data import Dataset


class ExperimentDataManager:
    """
    Manages the lifecycle of data for experiments.

    Responsibilities:
    - Verifying existence of feature artifacts.
    - Creating efficient memory-mapped views of data for parallel processing.
    - Consolidating distributed checkpoint results into final artifacts.
    """

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def artifacts_exist(self, dataset: Dataset) -> bool:
        """Checks if the necessary feature engineering artifacts exist."""
        paths = self.ctx.get_feature_paths(dataset.id)
        exists = paths["X"].exists() and paths["y"].exists()

        if not exists:
            self.ctx.logger.warning(f"Artifacts not found for {dataset.display_name}.")

        return exists

    def get_dataset_size_gb(self, dataset: Dataset) -> float:
        """Estimates the size of the raw dataset in GB for job calculation."""
        raw_path = self.ctx.get_raw_data_path(dataset.id)
        if raw_path.exists():
            return raw_path.stat().st_size / (1024**3)
        return 1.0  # Default fallback

    @contextmanager
    def feature_context(self, dataset: Dataset) -> Generator[tuple[str, str], None, None]:
        """
        Context manager that prepares data for parallel access.

        1. Loads Parquet files into memory.
        2. Dumps them to a temporary memory-mapped file (joblib format).
        3. Yields the paths to these memory maps.
        4. Cleans up temporary files and forces garbage collection on exit.

        Yields:
            tuple[str, str]: Paths to (X_mmap, y_mmap).
        """
        paths = self.ctx.get_feature_paths(dataset.id)
        x_path, y_path = paths["X"], paths["y"]

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Data missing for {dataset.display_name}")

        self.ctx.logger.info(f"Loading data for {dataset.display_name}...")

        # Load original data into RAM
        # Note: We convert Polars to Pandas here because joblib/sklearn
        # integration with numpy arrays via Pandas is currently more robust for memmapping.
        X_df = pl.read_parquet(x_path).to_pandas()
        y_df = pl.read_parquet(y_path).to_pandas().iloc[:, 0]

        # Create a temporary directory for memory mapping
        with tempfile.TemporaryDirectory() as temp_dir:
            X_mmap_path = os.path.join(temp_dir, "X.mmap")
            y_mmap_path = os.path.join(temp_dir, "y.mmap")

            # Dump data using joblib (optimized for numpy persistence)
            joblib.dump(X_df.to_numpy(), X_mmap_path)
            joblib.dump(y_df.to_numpy(), y_mmap_path)

            # Quick verification load to ensure integrity
            _ = joblib.load(X_mmap_path, mmap_mode="r")

            # aggressively free the original RAM before yielding control
            del X_df, y_df
            gc.collect()

            self.ctx.logger.info(f"Data memory-mapped for {dataset.display_name}")

            try:
                yield X_mmap_path, y_mmap_path
            finally:
                # Context exit: temp_dir cleanup handles file deletion,
                # but explicit GC helps ensure file handles are released.
                gc.collect()

    def consolidate_results(self, dataset: Dataset) -> None:
        """
        Aggregates individual task checkpoints into a single results file.

        It looks for all .parquet files in the dataset's checkpoint directory,
        concatenates them, and saves the result to the consolidated path.
        """
        # We use a sample checkpoint path to find the parent directory
        # logic relies on structure: .../checkpoints/{dataset_id}/...
        sample_ckpt = self.ctx.get_checkpoint_path(dataset.id, "x", "y", 0)
        ckpt_dir = sample_ckpt.parent

        all_files = list(ckpt_dir.glob("*.parquet"))

        if not all_files:
            self.ctx.logger.warning(f"No results found for {dataset.display_name}")
            return

        self.ctx.logger.info(
            f"Consolidating {len(all_files)} results for {dataset.display_name}..."
        )

        frames = []
        for fp in all_files:
            try:
                frames.append(pd.read_parquet(fp))
            except Exception as e:
                self.ctx.logger.warning(f"Failed to read checkpoint {fp.name}: {e}")

        if frames:
            df_final = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
            final_output = self.ctx.get_consolidated_results_path(dataset.id)

            # Ensure results directory exists
            final_output.parent.mkdir(parents=True, exist_ok=True)

            df_final.to_parquet(final_output)
            self.ctx.logger.success(f"Saved consolidated results to {final_output}")
        else:
            self.ctx.logger.warning("No valid data frames could be loaded.")
