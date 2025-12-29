"""Protocol definitions for training pipeline components.

This module defines the interfaces (protocols) for the training pipeline stages:
task generation, data loading, training execution, and results persistence.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Protocol, runtime_checkable

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.model_versioning import ModelVersioningService


@dataclass(frozen=True, slots=True)
class ExperimentTask:
    """Represents a single experiment task to be executed.

    Attributes:
        dataset: The dataset to train on.
        model_type: The model type to use.
        technique: The technique for handling class imbalance.
        seed: Random seed for reproducibility.
    """

    dataset: Dataset
    model_type: ModelType
    technique: Technique
    seed: int


@runtime_checkable
class TaskGenerator(Protocol):
    """Protocol for generating experiment tasks."""

    def generate(
        self,
        datasets: list[Dataset],
        excluded_models: set[ModelType] | None = None,
    ) -> list[ExperimentTask]:
        """Generate all experiment tasks for the given datasets.

        Args:
            datasets: List of datasets to generate tasks for.
            excluded_models: Model types to exclude from generation.

        Returns:
            List of experiment tasks to execute.
        """
        ...


@runtime_checkable
class CheckpointUriProvider(Protocol):
    """Protocol for providing checkpoint URIs."""

    def get_checkpoint_uri(
        self,
        dataset_id: str,
        model_id: str,
        technique_id: str,
        seed: int,
    ) -> str:
        """Get the checkpoint URI for a specific task."""
        ...


@runtime_checkable
class ConsolidatedResultsUriProvider(Protocol):
    """Protocol for providing consolidated results URIs."""

    def get_consolidated_results_uri(self, dataset_id: str) -> str:
        """Get the URI for consolidated results."""
        ...


@runtime_checkable
class ModelVersioningProvider(Protocol):
    """Protocol for providing model versioning services."""

    def get_model_versioning_service(
        self,
        dataset_id: str,
        model_type: ModelType,
        technique: Technique,
    ) -> ModelVersioningService:
        """Get a model versioning service for the given configuration."""
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for providing memory-mapped data paths."""

    @contextmanager
    def feature_context(self, dataset: Dataset) -> Generator[tuple[str, str], None, None]:
        """Context manager providing paths to memory-mapped features.

        Yields:
            Tuple of (X_mmap_path, y_mmap_path).
        """
        ...

    def artifacts_exist(self, dataset: Dataset) -> bool:
        """Check if feature artifacts exist for the dataset."""
        ...

    def get_dataset_size_gb(self, dataset: Dataset) -> float:
        """Get the estimated size of the dataset in GB."""
        ...


# Type alias for experiment runner function
ExperimentRunner = Callable[..., str | None]


@runtime_checkable
class TrainingExecutor(Protocol):
    """Protocol for executing training tasks."""

    def execute(
        self,
        tasks: list[ExperimentTask],
        runner: ExperimentRunner,
        data_paths: tuple[str, str],
        config: Any,
        checkpoint_provider: CheckpointUriProvider,
        versioning_provider: ModelVersioningProvider,
    ) -> list[str | None]:
        """Execute the training tasks.

        Args:
            tasks: List of tasks to execute.
            runner: The function to run for each task.
            data_paths: Tuple of (X_mmap_path, y_mmap_path).
            config: Experiment configuration.
            checkpoint_provider: Provider for checkpoint URIs.
            versioning_provider: Provider for model versioning services.

        Returns:
            List of task identifiers for completed tasks.
        """
        ...


@runtime_checkable
class ResultsConsolidator(Protocol):
    """Protocol for consolidating experiment results."""

    def consolidate(self, dataset: Dataset) -> str | None:
        """Consolidate checkpoint results for a dataset.

        Args:
            dataset: The dataset to consolidate results for.

        Returns:
            URI to the consolidated results file, or None if no results.
        """
        ...


# Backwards compatibility aliases (deprecated)
CheckpointPathProvider = CheckpointUriProvider
ConsolidatedResultsPathProvider = ConsolidatedResultsUriProvider
