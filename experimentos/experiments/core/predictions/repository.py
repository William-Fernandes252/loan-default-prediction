"""Defines the repository for managing model predictions related to experiments."""

from dataclasses import dataclass
from typing import Iterator, NamedTuple, Protocol

import polars as pl

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique


class ExperimentCombination(NamedTuple):
    """Represents a unique experiment combination."""

    dataset: Dataset
    model_type: ModelType
    technique: Technique
    seed: int


@dataclass(frozen=True, slots=True)
class ModelPredictions:
    """Represents the results of an experiment on a dataset, for a specific model and technique.

    The results are stored as a Polars `LazyFrame` that contains the features, along with their target (from column `"target"`) values and predictions (from column `"prediction"`) values.
    """

    execution_id: str
    seed: int
    dataset: Dataset
    model_type: ModelType
    technique: Technique
    predictions: pl.LazyFrame


type ModelPredictionsResults = Iterator[ModelPredictions]
"""Retrieved model predictions results."""


class ModelPredictionsRepository(Protocol):
    def get_completed_combinations(self, execution_id: str) -> set[ExperimentCombination]:
        """Get all completed experiment combinations for a given execution.

        Args:
            execution_id (str): The execution identifier to query.

        Returns:
            set[ExperimentCombination]: A set of completed combinations.
        """
        ...

    def get_latest_predictions_for_experiment(
        self, dataset: Dataset
    ) -> ModelPredictionsResults | None:
        """Fetches the latest experiment results for a given dataset.

        Args:
            dataset (Dataset): The dataset for which to fetch results.

        Returns:
            ModelPredictionsResults | None: The latest results as an iterator of model predictions, one for each seed, or `None` if no results exist.

        Raises:
            Exception: If there is an error retrieving the results.
        """
        ...

    def save_predictions(
        self,
        *,
        execution_id: str,
        seed: int,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        predictions: pl.LazyFrame,
    ) -> None:
        """Saves the model predictions for a given experiment.

        Args:
            execution_id (str): The unique identifier for the experiment execution.
            dataset (Dataset): The dataset associated with the experiment.
            model_type (ModelType): The type of model used.
            technique (Technique): The technique applied.
            seed (int): The random seed used during training.
            predictions (pl.LazyFrame): The predictions to save.

        Raises:
            Exception: If there is an error saving the predictions.
        """
        ...
