"""Define abstractions and base implementation for analysis pipelines."""

from abc import ABC, abstractmethod
from typing import (
    BinaryIO,
    Callable,
    ClassVar,
    Protocol,
)

from experiments.core.data import Dataset
from experiments.lib.pipelines import Pipeline, TaskResult, TaskStatus
from experiments.pipelines.analysis.pipeline import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineState,
    AnalysisPipelineTask,
    AnalysisPipelineTaskResult,
)
from experiments.pipelines.analysis.tasks.common import load_experiment_results


class AnalysisArtifactRepository(Protocol):
    def save_analysis_artifact(
        self,
        dataset: Dataset,
        analysis_name: str,
        artifact_data: BinaryIO,
        locale: str = "en_US",
    ) -> None:
        """Saves an analysis artifact for a given experiment.

        Args:
            dataset (Dataset): The dataset associated with the experiment.
            analysis_name (str): The name of the analysis artifact to be saved.
            artifact_data (BinaryIO): The binary data of the artifact.
            locale (str): The locale for the artifact (default: 'en_US').

        Raises:
            Exception: If there is an error saving the artifact.
        """
        ...

    def artifact_exists(
        self,
        dataset: Dataset,
        analysis_name: str,
        locale: str = "en_US",
    ) -> bool:
        """Checks if an analysis artifact exists in the repository.

        Args:
            dataset (Dataset): The dataset associated with the experiment.
            analysis_name (str): The name of the analysis artifact to check.

        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        ...


type ArtifactGenerator[T] = Callable[[T, AnalysisPipelineContext], BinaryIO]
"""Callable that generates an artifact from the analysis result and context."""


def export_analysis_artifact(
    state: AnalysisPipelineState, context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult:
    """Export analysis artifact to the results repository."""
    if "artifact" not in state:
        return TaskResult(state, TaskStatus.FAILURE, "No artifact to export.")

    artifact = state["artifact"]
    context.analysis_artifacts_repository.save_analysis_artifact(
        context.dataset,
        context.analysis_name,
        artifact,
        locale=context.locale.value,
    )
    return TaskResult(state, TaskStatus.SUCCESS, "Exported artifact.")


def run_if_artifact_not_exists(state: AnalysisPipelineState, context: AnalysisPipelineContext):
    if state.get("already_exists", True) and not context.force_overwrite:
        return False, "Artifact already exists; skipping step."
    return True, None


class AnalysisPipelineFactory[T](ABC):
    """Abstract factory for creating analysis pipelines."""

    _NAME: ClassVar[str] = "Analysis"
    """The name of the analysis."""

    def get_pipeline_name(self, params: dict[str, str]) -> str:
        """Get the name of the analysis pipeline for logging."""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self._NAME}[{params_str}]"

    def __init__(
        self,
        analysis_artifacts_repository: AnalysisArtifactRepository,
    ) -> None:
        self._analysis_artifacts_repository = analysis_artifacts_repository

    @abstractmethod
    def _add_analysis_steps(self, pipeline: AnalysisPipeline[T]) -> None:
        """Add analysis steps to the pipeline.

        The analysis steps will be added directly after the result loading step.
        They should populate the `result_data` key in the pipeline state.
        """
        ...

    def _add_artifact_generation_step(
        self,
        pipeline: AnalysisPipeline[T],
        artifact_generator: ArtifactGenerator[T],
        step_name: str,
    ) -> None:
        """Add artifact generation step to the pipeline."""
        pipeline.add_step(
            name=step_name,
            task=lambda state, context: TaskResult(
                {**state, "artifact": artifact_generator(state["result_data"], context)},
                TaskStatus.SUCCESS,
                f"Generated artifact using {step_name}.",
            ),
        )

    def create_pipeline(
        self,
        name: str,
        artifact_generator: ArtifactGenerator[T],
    ) -> AnalysisPipeline[T]:
        """Create and return an analysis pipeline.

        Returns:
            An instance of AnalysisPipeline.
        """

        pipeline = Pipeline[AnalysisPipelineState[T], AnalysisPipelineContext](name=name)

        pipeline.add_step(
            name="CheckArtifactExists",
            task=self.__create_check_artifact_step(),
        )
        pipeline.add_conditional_step(
            name="LoadResults",
            task=load_experiment_results,
            condition=run_if_artifact_not_exists,
        )
        self._add_analysis_steps(pipeline)
        self._add_artifact_generation_step(pipeline, artifact_generator, "GenerateArtifact")
        pipeline.add_conditional_step(
            name="ExportArtifact",
            task=export_analysis_artifact,
            condition=run_if_artifact_not_exists,
        )

        return pipeline

    def __create_check_artifact_step(self) -> AnalysisPipelineTask[T]:
        """Create a step to check if the artifact already exists."""

        def check(
            state: AnalysisPipelineState,
            context: AnalysisPipelineContext,
        ) -> AnalysisPipelineTaskResult:
            state["already_exists"] = context.analysis_artifacts_repository.artifact_exists(
                context.dataset,
                self._NAME,
            )
            return TaskResult(
                state,
                TaskStatus.SUCCESS,
                "Artifact already exists."
                if state["already_exists"]
                else "Artifact does not exist.",
            )

        return check
