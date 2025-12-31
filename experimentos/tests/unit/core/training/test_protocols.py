"""Tests for experiments.core.training.protocols module."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training.protocols import (
    CheckpointUriProvider,
    DataProvider,
    ExperimentTask,
    ModelVersioningProvider,
    ResultsConsolidator,
    TaskGenerator,
    TrainingExecutor,
)


class DescribeExperimentTask:
    """Tests for ExperimentTask dataclass."""

    def it_stores_all_task_fields(self) -> None:
        """Verify ExperimentTask stores all fields."""
        task = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        assert task.dataset == Dataset.TAIWAN_CREDIT
        assert task.model_type == ModelType.RANDOM_FOREST
        assert task.technique == Technique.BASELINE
        assert task.seed == 42

    def it_is_frozen(self) -> None:
        """Verify ExperimentTask is immutable."""
        task = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        with pytest.raises(AttributeError):
            task.seed = 100  # type: ignore[misc]

    def it_is_hashable(self) -> None:
        """Verify ExperimentTask can be used in sets and dicts."""
        task1 = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )
        task2 = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        task_set = {task1, task2}
        assert len(task_set) == 1

    def it_supports_equality(self) -> None:
        """Verify ExperimentTask supports equality comparison."""
        task1 = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )
        task2 = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )
        task3 = ExperimentTask(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            seed=42,
        )

        assert task1 == task2
        assert task1 != task3


class DescribeTaskGeneratorProtocol:
    """Tests for TaskGenerator protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify TaskGenerator can be checked at runtime."""

        class ValidGenerator:
            def generate(
                self,
                datasets: list[Dataset],
                excluded_models: set[ModelType] | None = None,
            ) -> list[ExperimentTask]:
                return []

        generator = ValidGenerator()
        assert isinstance(generator, TaskGenerator)

    def it_rejects_non_conforming_classes(self) -> None:
        """Verify non-conforming classes are rejected."""

        class InvalidGenerator:
            def wrong_method(self) -> None:
                pass

        generator = InvalidGenerator()
        assert not isinstance(generator, TaskGenerator)


class DescribeCheckpointUriProviderProtocol:
    """Tests for CheckpointUriProvider protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify CheckpointUriProvider can be checked at runtime."""

        class ValidProvider:
            def get_checkpoint_uri(
                self,
                dataset_id: str,
                model_id: str,
                technique_id: str,
                seed: int,
            ) -> str:
                return "/checkpoints/test.parquet"

        provider = ValidProvider()
        assert isinstance(provider, CheckpointUriProvider)


class DescribeConsolidatedResultsUriProviderProtocol:
    """Tests for ConsolidatedResultsUriProvider protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify ConsolidatedResultsUriProvider can be checked at runtime."""

        class ValidProvider:
            def get_consolidated_results_uri(self, dataset_id: str) -> str:
                return "/results/test.parquet"

        provider = ValidProvider()
        from experiments.core.training.protocols import ConsolidatedResultsUriProvider

        assert isinstance(provider, ConsolidatedResultsUriProvider)


class DescribeModelVersioningProviderProtocol:
    """Tests for ModelVersioningProvider protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify ModelVersioningProvider can be checked at runtime."""
        from unittest.mock import MagicMock

        class ValidProvider:
            def get_model_versioning_service(
                self,
                dataset_id: str,
                model_type: ModelType,
                technique: Technique,
            ):
                return MagicMock()

        provider = ValidProvider()
        assert isinstance(provider, ModelVersioningProvider)


class DescribeDataProviderProtocol:
    """Tests for DataProvider protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify DataProvider can be checked at runtime."""

        class ValidProvider:
            @contextmanager
            def feature_context(self, dataset: Dataset) -> Generator[tuple[str, str], None, None]:
                yield ("/path/X.joblib", "/path/y.joblib")

            def artifacts_exist(self, dataset: Dataset) -> bool:
                return True

            def get_dataset_size_gb(self, dataset: Dataset) -> float:
                return 1.0

        provider = ValidProvider()
        assert isinstance(provider, DataProvider)


class DescribeTrainingExecutorProtocol:
    """Tests for TrainingExecutor protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify TrainingExecutor can be checked at runtime."""

        class ValidExecutor:
            def execute(
                self,
                tasks: list[ExperimentTask],
                runner: object,
                data_paths: tuple[str, str],
                config: object,
                checkpoint_provider: object,
                versioning_provider: object,
            ) -> list[str | None]:
                return []

        executor = ValidExecutor()
        assert isinstance(executor, TrainingExecutor)


class DescribeResultsConsolidatorProtocol:
    """Tests for ResultsConsolidator protocol."""

    def it_is_runtime_checkable(self) -> None:
        """Verify ResultsConsolidator can be checked at runtime."""

        class ValidConsolidator:
            def consolidate(self, dataset: Dataset) -> Path | None:
                return None

        consolidator = ValidConsolidator()
        assert isinstance(consolidator, ResultsConsolidator)
