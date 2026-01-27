"""Tests for resource_calculator service."""

from unittest.mock import patch

from experiments.services.resource_calculator import ResourceCalculator


class DescribeResourceCalculatorInit:
    def it_uses_default_safety_factor(self) -> None:
        calculator = ResourceCalculator()

        assert calculator.safety_factor == 3.5

    def it_accepts_custom_safety_factor(self) -> None:
        calculator = ResourceCalculator(safety_factor=2.0)

        assert calculator.safety_factor == 2.0


class DescribeComputeSafeJobs:
    def it_returns_at_least_one_job(self, resource_calculator: ResourceCalculator) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_mem.return_value.available = 1 * 1024**3  # 1 GB
            mock_cpu.return_value = 8

            result = resource_calculator.compute_safe_jobs(dataset_size_gb=100.0)

            assert result >= 1

    def it_respects_cpu_count_as_upper_bound(
        self, resource_calculator: ResourceCalculator
    ) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_mem.return_value.available = 1000 * 1024**3  # 1 TB
            mock_cpu.return_value = 4

            result = resource_calculator.compute_safe_jobs(dataset_size_gb=0.1)

            assert result <= 4

    def it_scales_jobs_with_available_memory(
        self, resource_calculator: ResourceCalculator
    ) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_cpu.return_value = 16

            mock_mem.return_value.available = 16 * 1024**3
            jobs_16gb = resource_calculator.compute_safe_jobs(dataset_size_gb=1.0)

            mock_mem.return_value.available = 32 * 1024**3
            jobs_32gb = resource_calculator.compute_safe_jobs(dataset_size_gb=1.0)

            assert jobs_32gb >= jobs_16gb

    def it_reduces_jobs_for_larger_datasets(self, resource_calculator: ResourceCalculator) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_mem.return_value.available = 16 * 1024**3
            mock_cpu.return_value = 16

            jobs_small = resource_calculator.compute_safe_jobs(dataset_size_gb=0.5)
            jobs_large = resource_calculator.compute_safe_jobs(dataset_size_gb=5.0)

            assert jobs_small >= jobs_large

    def it_handles_very_small_datasets(self, resource_calculator: ResourceCalculator) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_mem.return_value.available = 8 * 1024**3
            mock_cpu.return_value = 4

            result = resource_calculator.compute_safe_jobs(dataset_size_gb=0.001)

            assert result >= 1

    def it_handles_cpu_count_returning_none(self, resource_calculator: ResourceCalculator) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_mem.return_value.available = 8 * 1024**3
            mock_cpu.return_value = None

            result = resource_calculator.compute_safe_jobs(dataset_size_gb=1.0)

            assert result >= 1

    def it_applies_train_fraction_to_memory_estimate(
        self, resource_calculator: ResourceCalculator
    ) -> None:
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            mock_mem.return_value.available = 16 * 1024**3
            mock_cpu.return_value = 16

            jobs_70pct = resource_calculator.compute_safe_jobs(
                dataset_size_gb=2.0, train_fraction=0.70
            )
            jobs_50pct = resource_calculator.compute_safe_jobs(
                dataset_size_gb=2.0, train_fraction=0.50
            )

            assert jobs_50pct >= jobs_70pct
