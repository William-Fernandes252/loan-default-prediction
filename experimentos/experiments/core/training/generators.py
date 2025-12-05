"""Task generation implementations for the training pipeline."""

from dataclasses import dataclass, field

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training.protocols import ExperimentTask


@dataclass
class TaskGeneratorConfig:
    """Configuration for task generation.

    Attributes:
        num_seeds: Number of random seeds to use.
        excluded_models: Model types to exclude.
    """

    num_seeds: int = 30
    excluded_models: set[ModelType] = field(default_factory=set)


class ExperimentTaskGenerator:
    """Generates experiment tasks for all valid model/technique/seed combinations.

    This generator creates tasks while respecting:
    - Excluded model types
    - Invalid technique/model combinations (e.g., CS_SVM only for SVM)
    """

    def __init__(self, config: TaskGeneratorConfig) -> None:
        """Initialize the generator.

        Args:
            config: Configuration for task generation.
        """
        self._config = config

    def _is_valid_combination(self, model_type: ModelType, technique: Technique) -> bool:
        """Check if a model/technique combination is valid."""
        # CS_SVM is only valid for SVM models
        if technique == Technique.CS_SVM and model_type != ModelType.SVM:
            return False
        return True

    def generate(
        self,
        datasets: list[Dataset],
        excluded_models: set[ModelType] | None = None,
    ) -> list[ExperimentTask]:
        """Generate all experiment tasks for the given datasets.

        Args:
            datasets: List of datasets to generate tasks for.
            excluded_models: Additional model types to exclude.

        Returns:
            List of experiment tasks to execute.
        """
        # Combine configured exclusions with runtime exclusions
        all_excluded = self._config.excluded_models.copy()
        if excluded_models:
            all_excluded = all_excluded | excluded_models

        available_models = [m for m in ModelType if m not in all_excluded]
        if not available_models:
            return []

        tasks: list[ExperimentTask] = []

        for dataset in datasets:
            for seed in range(self._config.num_seeds):
                for model_type in available_models:
                    for technique in Technique:
                        if not self._is_valid_combination(model_type, technique):
                            continue

                        tasks.append(
                            ExperimentTask(
                                dataset=dataset,
                                model_type=model_type,
                                technique=technique,
                                seed=seed,
                            )
                        )

        return tasks

    def generate_single(
        self,
        dataset: Dataset,
        model_type: ModelType,
        technique: Technique,
        seed: int,
    ) -> ExperimentTask | None:
        """Generate a single experiment task.

        Args:
            dataset: The dataset to use.
            model_type: The model type.
            technique: The technique to use.
            seed: The random seed.

        Returns:
            The experiment task, or None if the combination is invalid.
        """
        if not self._is_valid_combination(model_type, technique):
            return None

        return ExperimentTask(
            dataset=dataset,
            model_type=model_type,
            technique=technique,
            seed=seed,
        )


__all__ = [
    "TaskGeneratorConfig",
    "ExperimentTaskGenerator",
]
