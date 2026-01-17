"""Resource calculation service for parallel processing.

This module provides a ResourceCalculator class that encapsulates
the logic for determining safe parallelization levels based on
available system resources.
"""

import psutil


class ResourceCalculator:
    """Calculates safe resource allocation for parallel processing.

    This class determines the appropriate number of parallel jobs
    based on available RAM and estimated memory requirements per worker.

    Attributes:
        safety_factor: Multiplier for peak memory usage estimation.
    """

    def __init__(self, safety_factor: float = 3.5) -> None:
        """Initialize the ResourceCalculator.

        Args:
            safety_factor: Multiplier for peak memory usage. Higher values
                are more conservative (fewer jobs). Default is 3.5.
        """
        self._safety_factor = safety_factor

    def compute_safe_jobs(self, dataset_size_gb: float, train_fraction: float = 0.70) -> int:
        """Calculate safe number of parallel jobs based on available RAM.

        The calculation considers:
        1. Available RAM on the system
        2. Estimated peak memory usage per worker
        3. CPU count as an upper bound

        Args:
            dataset_size_gb: Size of the dataset in GB.
            train_fraction: Fraction of the dataset used for training.
        Returns:
            Safe number of parallel jobs (at least 1, at most CPU count).
        """
        # Get available RAM in GB
        available_ram_gb = psutil.virtual_memory().available / (1024**3)

        # Estimate peak memory required per worker
        # Only ~70% of data is loaded for training in splits
        train_size_gb = dataset_size_gb * train_fraction
        peak_memory_per_worker = train_size_gb * self._safety_factor

        # Avoid division by zero for tiny datasets
        if peak_memory_per_worker < 0.1:
            peak_memory_per_worker = 0.1

        # Calculate jobs
        safe_jobs = int(available_ram_gb // peak_memory_per_worker)

        # Ensure at least 1 job and don't exceed CPU count
        cpu_count = psutil.cpu_count(logical=False) or 1
        return max(1, min(safe_jobs, cpu_count))

    @property
    def safety_factor(self) -> float:
        """Get the current safety factor."""
        return self._safety_factor
