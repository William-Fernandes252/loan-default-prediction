"""Implementation of the model predictions repository using a storage backend."""

from collections.abc import Iterator
from dataclasses import dataclass
import re
from typing import TypedDict

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import ModelPredictions, ModelPredictionsResults
from experiments.storage import Storage
from experiments.storage.interface import FileInfo


class _ParsedPredictionsKey(TypedDict):
    """Parsed components from a predictions storage key."""

    dataset: str
    model_type: str
    technique: str
    seed: int


@dataclass(frozen=True, slots=True)
class ModelPredictionsStorageLayout:
    """Layout for storing model predictions in the storage backend."""

    predictions_key_template: str = (
        "predictions/{dataset}/{model_type}/{technique}/seed_{seed}.parquet"
    )
    predictions_prefix: str = "predictions/"

    def get_predictions_key(
        self,
        dataset: str,
        model_type: str,
        technique: str,
        seed: int,
    ) -> str:
        """Generate the storage key for model predictions.

        Args:
            dataset: The dataset the model was trained on.
            model_type: The type of the model.
            technique: The technique used for training.
            seed: The random seed used during training.

        Returns:
            str: The storage key for the model predictions.
        """
        return self.predictions_key_template.format(
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
            A dictionary with dataset, model_type, technique, and seed,
            or None if the key doesn't match the expected pattern.
        """
        pattern = r"^predictions/([^/]+)/([^/]+)/([^/]+)/seed_(\d+)\.parquet$"
        match = re.match(pattern, key)
        if not match:
            return None
        return _ParsedPredictionsKey(
            dataset=match.group(1),
            model_type=match.group(2),
            technique=match.group(3),
            seed=int(match.group(4)),
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

    def save_predictions(
        self,
        predictions: ModelPredictions,
        *,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        seed: int,
    ) -> None:
        """Saves model predictions to the storage backend.

        Args:
            predictions: The model predictions to save.
            dataset: The dataset the model was trained on.
            model_type: The type of the model.
            technique: The technique used for training.
            seed: The random seed used during training.
        """
        key = self._layout.get_predictions_key(
            dataset=dataset.value,
            model_type=model_type.value,
            technique=technique.value,
            seed=seed,
        )
        self._storage.sink_parquet(predictions.predictions, key)

    def get_latest_predictions_for_experiment(
        self, dataset: Dataset
    ) -> ModelPredictionsResults | None:
        """Fetches the latest experiment results for a given dataset.

        Args:
            dataset: The dataset for which to fetch results.

        Returns:
            ModelPredictionsResults: An iterator of model predictions,
            or None if no results exist.
        """
        prefix = f"{self._layout.predictions_prefix}{dataset.value}/"
        files = list(self._storage.list_files(prefix, "*.parquet"))

        if not files:
            return None

        return self._iter_predictions(dataset, files)

    def _iter_predictions(
        self, dataset: Dataset, files: list[FileInfo]
    ) -> Iterator[ModelPredictions]:
        """Iterate over prediction files and yield ModelPredictions.

        Args:
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
                    dataset=dataset,
                    model_type=model_type,
                    technique=technique,
                    predictions=predictions_lf,
                )
            except ValueError:
                # Skip files that don't match valid enum values
                continue
