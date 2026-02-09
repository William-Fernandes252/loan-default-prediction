"""Implementation of the model predictions repository using a storage backend."""

from collections.abc import Iterator
from dataclasses import dataclass
import re
from typing import TypedDict

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import (
    ExperimentCombination,
    ModelPredictions,
    ModelPredictionsResults,
    RawPredictions,
    to_lazy_frame,
)
from experiments.storage import Storage
from experiments.storage.interface import FileInfo


class ExecutionNotFoundError(Exception):
    """Raised when a specified execution ID does not exist."""

    def __init__(self, execution_id: str, dataset: str | None = None):
        self.execution_id = execution_id
        self.dataset = dataset
        if dataset:
            message = f"Execution '{execution_id}' not found for dataset '{dataset}'"
        else:
            message = f"Execution '{execution_id}' not found"
        super().__init__(message)


class _ParsedPredictionsKey(TypedDict):
    """Parsed components from a predictions storage key."""

    execution_id: str
    dataset: str
    model_type: str
    technique: str
    seed: int


@dataclass(frozen=True, slots=True)
class ModelPredictionsStorageLayout:
    """Layout for storing model predictions in the storage backend."""

    predictions_key_template: str = (
        "predictions/{execution_id}/{dataset}/{model_type}/{technique}/seed_{seed}.parquet"
    )
    predictions_prefix: str = "predictions/"

    def get_predictions_key(
        self,
        execution_id: str,
        dataset: str,
        model_type: str,
        technique: str,
        seed: int,
    ) -> str:
        """Generate the storage key for model predictions.

        Args:
            execution_id: The unique identifier for the experiment execution.
            dataset: The dataset the model was trained on.
            model_type: The type of the model.
            technique: The technique used for training.
            seed: The random seed used during training.

        Returns:
            str: The storage key for the model predictions.
        """
        return self.predictions_key_template.format(
            execution_id=execution_id,
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            seed=seed,
        )

    def parse_predictions_key(self, key: str) -> _ParsedPredictionsKey | None:
        """Parse a predictions key and extract its components.

        Args:
            key: The storage key to parse.

        Returns:
            _ParsedPredictionsKey | None: The parsed components, or `None` if the key does not match the expected format.
        """
        pattern = r"^predictions/([^/]+)/([^/]+)/([^/]+)/([^/]+)/seed_(\d+)\.parquet$"
        match = re.match(pattern, key)
        if not match:
            return None
        return _ParsedPredictionsKey(
            execution_id=match.group(1),
            dataset=match.group(2),
            model_type=match.group(3),
            technique=match.group(4),
            seed=int(match.group(5)),
        )


class ModelPredictionsStorageRepository:
    """Model predictions repository implementation using a storage backend."""

    def __init__(
        self,
        storage: Storage,
        layout: ModelPredictionsStorageLayout | None = None,
    ):
        self._storage = storage
        self._layout = layout or ModelPredictionsStorageLayout()

    def get_completed_combinations(self, execution_id: str) -> set[ExperimentCombination]:
        """Get all completed experiment combinations for a given execution.

        Args:
            execution_id: The execution identifier to query.

        Returns:
            A set of completed (dataset, model_type, technique, seed) combinations.
        """
        prefix = f"{self._layout.predictions_prefix}{execution_id}/"
        completed: set[ExperimentCombination] = set()

        for file_info in self._storage.list_files(prefix, "*.parquet"):
            parsed = self._layout.parse_predictions_key(file_info.key)
            if parsed and parsed["execution_id"] == execution_id:
                try:
                    completed.add(
                        ExperimentCombination(
                            dataset=Dataset(parsed["dataset"]),
                            model_type=ModelType(parsed["model_type"]),
                            technique=Technique(parsed["technique"]),
                            seed=parsed["seed"],
                        )
                    )
                except ValueError:
                    # Skip invalid enum values
                    continue

        return completed

    def get_latest_execution_id(
        self,
        datasets: list[Dataset] | None = None,
    ) -> str | None:
        """Find the most recent execution ID, optionally filtered by datasets.

        This method lists all prediction files, extracts execution IDs, and returns
        the latest one based on UUID7's time-sortable property (higher = more recent).

        Args:
            datasets: Optional list of datasets to filter by. If provided, only returns
                     execution IDs that have predictions for at least one of these datasets.

        Returns:
            The latest execution ID, or None if no executions are found.

        Example:
            >>> repo.get_latest_execution_id([Dataset.TAIWAN_CREDIT])
            "01943abc-1234-7000-8000-0123456789ab"
        """
        prefix = self._layout.predictions_prefix
        execution_ids: set[str] = set()

        for file_info in self._storage.list_files(prefix, "*.parquet"):
            parsed = self._layout.parse_predictions_key(file_info.key)
            if not parsed:
                continue

            # Filter by datasets if specified (any overlap)
            if datasets is not None:
                try:
                    file_dataset = Dataset(parsed["dataset"])
                    if file_dataset not in datasets:
                        continue
                except ValueError:
                    continue

            execution_ids.add(parsed["execution_id"])

        if not execution_ids:
            return None

        # UUID7 is time-sortable: higher string value = more recent
        return max(execution_ids)

    def save_predictions(
        self,
        *,
        execution_id: str,
        seed: int,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        predictions: RawPredictions,
    ) -> None:
        """Saves model predictions to the storage backend.

        Args:
            execution_id: The unique identifier for the experiment execution.
            seed: The random seed used during training.
            dataset: The dataset the model was trained on.
            model_type: The type of the model.
            technique: The technique used for training.
            predictions : The raw predictions to save.
        """
        key = self._layout.get_predictions_key(
            execution_id=execution_id,
            dataset=dataset.value,
            model_type=model_type.value,
            technique=technique.value,
            seed=seed,
        )
        self._storage.sink_parquet(to_lazy_frame(predictions), key)

    def get_latest_predictions_for_experiment(
        self, dataset: Dataset
    ) -> ModelPredictionsResults | None:
        """Fetches the latest experiment results for a given dataset.

        This finds the most recent execution_id (by sorting) and returns
        all predictions for that execution.

        Args:
            dataset: The dataset for which to fetch results.

        Returns:
            ModelPredictionsResults: An iterator of model predictions,
            or None if no results exist.
        """
        # List all prediction files for this dataset across all executions
        prefix = self._layout.predictions_prefix
        all_files = list(self._storage.list_files(prefix, "*.parquet"))

        # Filter files for the specified dataset and group by execution_id
        dataset_files: dict[str, list[FileInfo]] = {}
        for file_info in all_files:
            parsed = self._layout.parse_predictions_key(file_info.key)
            if parsed and parsed["dataset"] == dataset.value:
                exec_id = parsed["execution_id"]
                if exec_id not in dataset_files:
                    dataset_files[exec_id] = []
                dataset_files[exec_id].append(file_info)

        if not dataset_files:
            return None

        # Get the latest execution_id (uuid7 is time-sortable)
        latest_execution_id = max(dataset_files.keys())
        files = dataset_files[latest_execution_id]

        return self._iter_predictions(latest_execution_id, dataset, files)

    def get_predictions_for_execution(
        self, dataset: Dataset, execution_id: str
    ) -> ModelPredictionsResults | None:
        """Fetches experiment results for a specific execution ID.

        Args:
            dataset: The dataset for which to fetch results.
            execution_id: The specific execution ID to retrieve predictions for.

        Returns:
            ModelPredictionsResults: An iterator of model predictions,
            or None if no results exist for the given execution.

        Raises:
            ExecutionNotFoundError: If the execution ID does not exist.
        """
        # Build prefix for the specific execution and dataset
        prefix = f"{self._layout.predictions_prefix}{execution_id}/{dataset.value}/"
        files = list(self._storage.list_files(prefix, "*.parquet"))

        if not files:
            # Check if the execution exists at all (for any dataset)
            execution_prefix = f"{self._layout.predictions_prefix}{execution_id}/"
            any_files = list(self._storage.list_files(execution_prefix, "*.parquet"))
            if not any_files:
                raise ExecutionNotFoundError(execution_id)
            # Execution exists but no predictions for this dataset
            return None

        return self._iter_predictions(execution_id, dataset, files)

    def _iter_predictions(
        self, execution_id: str, dataset: Dataset, files: list[FileInfo]
    ) -> Iterator[ModelPredictions]:
        """Iterate over prediction files and yield ModelPredictions.

        Args:
            execution_id: The execution identifier for the predictions.
            dataset: The dataset associated with the predictions.
            files: List of file info objects to iterate over.

        Yields:
            ModelPredictions: The model predictions for each file.
        """
        for file_info in files:
            parsed = self._layout.parse_predictions_key(file_info.key)
            if not parsed:
                continue

            try:
                model_type = ModelType(parsed["model_type"])
                technique = Technique(parsed["technique"])
                predictions_lf = self._storage.scan_parquet(file_info.key)

                yield ModelPredictions(
                    execution_id=execution_id,
                    seed=parsed["seed"],
                    dataset=dataset,
                    model_type=model_type,
                    technique=technique,
                    predictions=predictions_lf,
                )
            except ValueError:
                # Skip files that don't match valid enum values
                continue
