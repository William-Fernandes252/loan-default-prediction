""" "Pipeline definitions for experiment results analysis."""

from dataclasses import dataclass
from typing import (
    Annotated,
    BinaryIO,
    Protocol,
    TypedDict,
)

from experiments.core.analysis.evaluation import (
    EvaluationMetrics,
    ModelPredictionsResults,
    ModelResultsEvaluator,
)
from experiments.core.data import Dataset
from experiments.core.predictions.repository import ModelPredictionsRepository
from experiments.lib.pipelines import Pipeline, Task, TaskResult


class AnalysisArtifactRepository(Protocol):
    def save_analysis_artifact(
        self,
        dataset: Dataset,
        analysis_name: str,
        artifact_data: bytes,
    ) -> None:
        """Saves an analysis artifact for a given experiment.

        Args:
            dataset (Dataset): The dataset associated with the experiment.
            analysis_name (str): The name of the analysis artifact to be saved.
            artifact_data (bytes): The binary data of the artifact.

        Raises:
            Exception: If there is an error saving the artifact.
        """
        ...

    def artifact_exists(
        self,
        dataset: Dataset,
        analysis_name: str,
    ) -> bool:
        """Checks if an analysis artifact exists in the repository.

        Args:
            dataset (Dataset): The dataset associated with the experiment.
            analysis_name (str): The name of the analysis artifact to check.

        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        ...


class AnalysisPipelineState[T](TypedDict, total=False):
    """State for the experiment results analysis pipeline."""

    model_predictions: Annotated[
        ModelPredictionsResults | None, "The experiment results data loaded from the repository."
    ]
    metrics: Annotated[EvaluationMetrics | None, "Computed analysis metrics."]
    already_exists: Annotated[bool, "Flag indicating if the analysis artifact already exists."]
    result_data: Annotated[T, "The output of the analysis pipeline. Can be an image, report, etc."]
    artifact: Annotated[BinaryIO, "The binary artifact generated from the analysis."]


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisPipelineContext:
    """Context for experiment results analysis pipeline.

    Attributes:
        dataset (Dataset): The dataset used in the experiment.
        analysis_name (str): The name of the analysis being performed.
        predictions_repository (ModelPredictionsRepository): The repository for fetching model predictions.
        results_evaluator (ModelResultsEvaluator): The evaluator for computing analysis metrics.
        analysis_artifacts_repository (AnalysisArtifactRepository): The repository for saving analysis artifacts.
        use_gpu (bool): Flag to indicate if GPU acceleration should be used.
        force_overwrite (bool): Flag to indicate if existing artifacts should be overwritten.
    """

    dataset: Dataset
    analysis_name: str
    predictions_repository: ModelPredictionsRepository
    results_evaluator: ModelResultsEvaluator
    analysis_artifacts_repository: AnalysisArtifactRepository
    use_gpu: bool
    force_overwrite: bool


type AnalysisPipelineTask[T] = Task[AnalysisPipelineState[T], AnalysisPipelineContext]
"""Type alias for tasks in the analysis pipeline."""


type AnalysisPipelineTaskResult[T] = TaskResult[AnalysisPipelineState[T]]


type AnalysisPipeline[T] = Pipeline[AnalysisPipelineState[T], AnalysisPipelineContext]
"""Type alias for the analysis pipeline."""
