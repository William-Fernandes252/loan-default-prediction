"""Tests for resource_calculator service."""

from unittest.mock import patch

import pytest

from experiments.services.resource_calculator import ResourceCalculator


class DescribeResourceCalculatorInit:
    def it_uses_default_safety_factor(self) -> None:
        calc = ResourceCalculator()

        assert calc.safety_factor == 3.5

    def it_accepts_custom_safety_factor(self) -> None:
        calc = ResourceCalculator(safety_factor=2.0)

        assert calc.safety_factor == 2.0


class DescribeComputeSafeJobs:
    @pytest.fixture
    def calculator(self) -> ResourceCalculator:
        return ResourceCalculator(safety_factor=3.5)

    def it_returns_at_least_one_job(self, calculator: ResourceCalculator) -> None:
        # Even with huge datasets and limited RAM, should return at least 1
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_mem.return_value.available = 1 * 1024**3  # 1 GB
            mock_cpu.return_value = 8

            jobs = calculator.compute_safe_jobs(dataset_size_gb=100.0)

            assert jobs >= 1

    def it_respects_cpu_count_as_upper_bound(self, calculator: ResourceCalculator) -> None:
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_mem.return_value.available = 1000 * 1024**3  # 1 TB (plenty of RAM)
            mock_cpu.return_value = 4

            jobs = calculator.compute_safe_jobs(dataset_size_gb=0.1)

            assert jobs <= 4

    def it_scales_jobs_with_available_memory(self, calculator: ResourceCalculator) -> None:
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_cpu.return_value = 16

            # More RAM = more jobs
            mock_mem.return_value.available = 16 * 1024**3  # 16 GB
            jobs_with_16gb = calculator.compute_safe_jobs(dataset_size_gb=1.0)

            mock_mem.return_value.available = 32 * 1024**3  # 32 GB
            jobs_with_32gb = calculator.compute_safe_jobs(dataset_size_gb=1.0)

            assert jobs_with_32gb >= jobs_with_16gb

    def it_reduces_jobs_for_larger_datasets(self, calculator: ResourceCalculator) -> None:
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_mem.return_value.available = 16 * 1024**3  # 16 GB
            mock_cpu.return_value = 16

            jobs_small = calculator.compute_safe_jobs(dataset_size_gb=0.5)
            jobs_large = calculator.compute_safe_jobs(dataset_size_gb=5.0)

            assert jobs_small >= jobs_large

    def it_handles_very_small_datasets(self, calculator: ResourceCalculator) -> None:
        # Very small datasets should not cause division by zero
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_mem.return_value.available = 8 * 1024**3  # 8 GB
            mock_cpu.return_value = 4

            jobs = calculator.compute_safe_jobs(dataset_size_gb=0.001)

            assert jobs >= 1

    def it_handles_cpu_count_returning_none(self, calculator: ResourceCalculator) -> None:
        # Some systems may return None for cpu_count
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_mem.return_value.available = 8 * 1024**3
            mock_cpu.return_value = None  # Fallback to 1

            jobs = calculator.compute_safe_jobs(dataset_size_gb=1.0)

            assert jobs >= 1

    def it_applies_train_fraction_to_memory_estimate(self, calculator: ResourceCalculator) -> None:
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_count") as mock_cpu:
            mock_mem.return_value.available = 16 * 1024**3  # 16 GB
            mock_cpu.return_value = 16

            # Lower train fraction means less memory per worker = more jobs
            jobs_70pct = calculator.compute_safe_jobs(dataset_size_gb=2.0, train_fraction=0.70)
            jobs_50pct = calculator.compute_safe_jobs(dataset_size_gb=2.0, train_fraction=0.50)

            assert jobs_50pct >= jobs_70pct
