"""Tests for experiments.services.resource_calculator module."""

from unittest.mock import MagicMock, patch

import pytest

from experiments.services.resource_calculator import ResourceCalculator


@pytest.fixture
def resource_calculator() -> ResourceCalculator:
    """Create a ResourceCalculator with default settings."""
    return ResourceCalculator()


@pytest.fixture
def custom_calculator() -> ResourceCalculator:
    """Create a ResourceCalculator with custom safety factor."""
    return ResourceCalculator(safety_factor=2.0)


class DescribeResourceCalculator:
    """Tests for ResourceCalculator class."""

    class DescribeInit:
        """Tests for __init__ method."""

        def it_uses_default_safety_factor(self) -> None:
            """Verify default safety factor is 3.5."""
            calc = ResourceCalculator()
            assert calc.safety_factor == 3.5

        def it_accepts_custom_safety_factor(self) -> None:
            """Verify custom safety factor is used."""
            calc = ResourceCalculator(safety_factor=5.0)
            assert calc.safety_factor == 5.0

    class DescribeSafetyFactor:
        """Tests for safety_factor property."""

        def it_returns_configured_value(
            self,
            custom_calculator: ResourceCalculator,
        ) -> None:
            """Verify returns the configured safety factor."""
            assert custom_calculator.safety_factor == 2.0

    class DescribeComputeSafeJobs:
        """Tests for compute_safe_jobs method."""

        def it_returns_at_least_one_job(
            self,
            resource_calculator: ResourceCalculator,
        ) -> None:
            """Verify returns at least 1 job even with limited RAM."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                # Simulate very limited RAM (0.5 GB) with large dataset
                mock_vm = MagicMock()
                mock_vm.available = 0.5 * (1024**3)  # 0.5 GB in bytes
                mock_psutil.virtual_memory.return_value = mock_vm
                mock_psutil.cpu_count.return_value = 8

                result = resource_calculator.compute_safe_jobs(dataset_size_gb=10.0)

                assert result >= 1

        def it_respects_cpu_count_limit(
            self,
            resource_calculator: ResourceCalculator,
        ) -> None:
            """Verify does not exceed CPU count."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                # Simulate abundant RAM (100 GB)
                mock_vm = MagicMock()
                mock_vm.available = 100 * (1024**3)  # 100 GB
                mock_psutil.virtual_memory.return_value = mock_vm
                mock_psutil.cpu_count.return_value = 4

                result = resource_calculator.compute_safe_jobs(dataset_size_gb=0.1)

                assert result <= 4

        def it_calculates_based_on_available_ram(
            self,
            resource_calculator: ResourceCalculator,
        ) -> None:
            """Verify calculation is based on available RAM."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                # Simulate 16 GB available RAM
                mock_vm = MagicMock()
                mock_vm.available = 16 * (1024**3)  # 16 GB
                mock_psutil.virtual_memory.return_value = mock_vm
                mock_psutil.cpu_count.return_value = 16

                # With 1 GB dataset and safety factor 3.5:
                # peak_per_worker = 1.0 * 0.70 * 3.5 = 2.45 GB
                # safe_jobs = 16 / 2.45 ≈ 6
                result = resource_calculator.compute_safe_jobs(dataset_size_gb=1.0)

                assert 5 <= result <= 7

        def it_handles_tiny_datasets(
            self,
            resource_calculator: ResourceCalculator,
        ) -> None:
            """Verify handles very small datasets without division by zero."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                mock_vm = MagicMock()
                mock_vm.available = 8 * (1024**3)  # 8 GB
                mock_psutil.virtual_memory.return_value = mock_vm
                mock_psutil.cpu_count.return_value = 4

                # Very tiny dataset (essentially 0)
                result = resource_calculator.compute_safe_jobs(dataset_size_gb=0.0001)

                # Should not raise and should return a reasonable value
                assert 1 <= result <= 4

        def it_uses_physical_cpu_count(
            self,
            resource_calculator: ResourceCalculator,
        ) -> None:
            """Verify uses physical (not logical) CPU count."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                mock_vm = MagicMock()
                mock_vm.available = 100 * (1024**3)
                mock_psutil.virtual_memory.return_value = mock_vm

                # physical cores = 4, logical = 8 (hyperthreading)
                mock_psutil.cpu_count.return_value = 4

                result = resource_calculator.compute_safe_jobs(dataset_size_gb=0.01)

                # Should use physical core count
                mock_psutil.cpu_count.assert_called_with(logical=False)
                assert result <= 4

        def it_handles_none_cpu_count(
            self,
            resource_calculator: ResourceCalculator,
        ) -> None:
            """Verify handles case when cpu_count returns None."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                mock_vm = MagicMock()
                mock_vm.available = 100 * (1024**3)
                mock_psutil.virtual_memory.return_value = mock_vm
                mock_psutil.cpu_count.return_value = None  # Can happen on some systems

                result = resource_calculator.compute_safe_jobs(dataset_size_gb=0.1)

                # Should default to at least 1
                assert result >= 1

        def it_scales_with_safety_factor(self) -> None:
            """Verify higher safety factor results in fewer jobs."""
            with patch("experiments.services.resource_calculator.psutil") as mock_psutil:
                mock_vm = MagicMock()
                mock_vm.available = 16 * (1024**3)
                mock_psutil.virtual_memory.return_value = mock_vm
                mock_psutil.cpu_count.return_value = 16

                conservative_calc = ResourceCalculator(safety_factor=5.0)
                aggressive_calc = ResourceCalculator(safety_factor=2.0)

                conservative_jobs = conservative_calc.compute_safe_jobs(dataset_size_gb=1.0)
                aggressive_jobs = aggressive_calc.compute_safe_jobs(dataset_size_gb=1.0)

                assert conservative_jobs < aggressive_jobs
