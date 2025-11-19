"""Utility functions for job management based on system resources."""

import psutil


def get_safe_jobs(dataset_size_gb: float, safety_factor: float = 3.5) -> int:
    """
    Calculates safe number of jobs based on available RAM.

    Args:
        dataset_size_gb: Size of the dataset in GB.
        safety_factor: Multiplier for peak memory usage (3.5x is conservative for RF/Bagging).
    """
    # Get available RAM in GB
    available_ram_gb = psutil.virtual_memory().available / (1024**3)

    # Estimate peak memory required per worker
    # Only 70% of data is loaded for training
    train_size_gb = dataset_size_gb * 0.70
    peak_memory_per_worker = train_size_gb * safety_factor

    # Calculate jobs
    safe_jobs = int(available_ram_gb // peak_memory_per_worker)

    # Ensure at least 1 job and don't exceed CPU count
    cpu_count = psutil.cpu_count(logical=False) or 1
    return max(1, min(safe_jobs, cpu_count))
