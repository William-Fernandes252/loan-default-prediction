"""Protocol definitions for experiment pipeline components.

This module defines the interfaces for the experiment pipeline stages:
data splitting, model training, evaluation, and persistence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
from sklearn.base import BaseEstimator

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Paths to memory-mapped data.

    Attributes:
        X_path: Path to memory-mapped feature data.
        y_path: Path to memory-mapped label data.
    """

    X_path: str
    y_path: str


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """Identifies a unique experiment.

    Attributes:
        dataset: The dataset being trained on.
        model_type: The model type to use.
        technique: The technique for handling class imbalance.
        seed: Random seed for reproducibility.
    """

    dataset: Dataset
    model_type: ModelType
    technique: Technique
    seed: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        cv_folds: Number of cross-validation folds.
        cost_grids: Cost grid configurations.
    """

    cv_folds: int
    cost_grids: list[Any]


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """Complete experiment context composed of focused parts.

    Attributes:
        identity: Experiment identity (dataset, model, technique, seed).
        data: Paths to memory-mapped data.
        config: Training hyperparameters.
        checkpoint_path: Path to save checkpoint results.
        discard_checkpoints: Whether to discard existing checkpoints.
    """

    identity: ExperimentIdentity
    data: DataPaths
    config: TrainingConfig
    checkpoint_path: Path
    discard_checkpoints: bool = False


@dataclass(slots=True)
class SplitData:
    """Result of data splitting.

    Attributes:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass(slots=True)
class TrainedModel:
    """Result of model training.

    Attributes:
        estimator: The trained estimator.
        best_params: The best hyperparameters found.
    """

    estimator: BaseEstimator
    best_params: dict[str, Any]


@dataclass(slots=True)
class EvaluationResult:
    """Result of model evaluation.

    Attributes:
        metrics: Dictionary of metric names to values.
    """

    metrics: dict[str, float | str]


@dataclass(slots=True)
class ExperimentResult:
    """Final result of an experiment run.

    Attributes:
        task_id: Unique identifier for the task.
        metrics: All metrics including metadata.
        model: The trained model (optional).
    """

    task_id: str | None
    metrics: dict[str, Any]
    model: BaseEstimator | None = None


@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for splitting data into train/test sets."""

    def split(
        self,
        X_mmap_path: str,
        y_mmap_path: str,
        seed: int,
        cv_folds: int,
    ) -> SplitData | None:
        """Split data into train and test sets.

        Args:
            X_mmap_path: Path to memory-mapped feature data.
            y_mmap_path: Path to memory-mapped label data.
            seed: Random seed for reproducibility.
            cv_folds: Number of CV folds (used for validation).

        Returns:
            SplitData if successful, None if validation fails.
        """
        ...


@runtime_checkable
class ModelTrainer(Protocol):
    """Protocol for training and optimizing models."""

    def train(
        self,
        data: SplitData,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        cv_folds: int,
        cost_grids: list[Any],
    ) -> TrainedModel:
        """Train and optimize a model.

        Args:
            data: The split training/test data.
            model_type: Type of model to train.
            technique: Technique for handling class imbalance.
            seed: Random seed for reproducibility.
            cv_folds: Number of cross-validation folds.
            cost_grids: Cost grid configurations.

        Returns:
            The trained model with best parameters.
        """
        ...


@runtime_checkable
class ModelEvaluator(Protocol):
    """Protocol for evaluating trained models."""

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> EvaluationResult:
        """Evaluate a trained model on test data.

        Args:
            model: The trained model.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Evaluation result with metrics.
        """
        ...


@runtime_checkable
class EstimatorFactory(Protocol):
    """Protocol for creating estimators and parameter grids."""

    def create_pipeline(
        self,
        model_type: ModelType,
        technique: Technique,
        seed: int,
    ) -> ImbPipeline:
        """Create a pipeline for the given model type and technique.

        Args:
            model_type: Type of model to create.
            technique: Technique for handling class imbalance.
            seed: Random seed for reproducibility.

        Returns:
            The configured pipeline.
        """
        ...

    def get_param_grid(
        self,
        model_type: ModelType,
        technique: Technique,
        cost_grids: list[Any],
    ) -> list[dict[str, Any]]:
        """Get parameter grid for the given model type and technique.

        Args:
            model_type: Type of model.
            technique: Technique for handling class imbalance.
            cost_grids: Cost grid configurations.

        Returns:
            Parameter grid for hyperparameter search.
        """
        ...


@runtime_checkable
class ExperimentPersister(Protocol):
    """Protocol for persisting experiment results."""

    def save_checkpoint(
        self,
        metrics: dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        """Save experiment results to a checkpoint file.

        Args:
            metrics: Metrics dictionary to save.
            checkpoint_path: Path to save the checkpoint.
        """
        ...

    def save_model(
        self,
        model: BaseEstimator,
        context: ExperimentContext,
    ) -> None:
        """Save a trained model.

        Args:
            model: The trained model to save.
            context: Experiment context for path resolution.
        """
        ...

    def checkpoint_exists(self, checkpoint_path: Path) -> bool:
        """Check if a checkpoint already exists.

        Args:
            checkpoint_path: Path to check.

        Returns:
            True if checkpoint exists.
        """
        ...

    def discard_checkpoint(self, checkpoint_path: Path) -> None:
        """Discard an existing checkpoint.

        Args:
            checkpoint_path: Path to discard.
        """
        ...


__all__ = [
    "DataPaths",
    "ExperimentIdentity",
    "TrainingConfig",
    "ExperimentContext",
    "SplitData",
    "TrainedModel",
    "EvaluationResult",
    "ExperimentResult",
    "DataSplitter",
    "ModelTrainer",
    "ModelEvaluator",
    "EstimatorFactory",
    "ExperimentPersister",
]
