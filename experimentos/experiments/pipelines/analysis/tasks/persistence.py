"""Persistence tasks for saving analysis artifacts.

These tasks handle the IO side-effects of persisting analysis results
to storage. They serialize DataFrames to LaTeX and Figures to PNG bytes.
"""

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from experiments.lib.pipelines.tasks import TaskResult, TaskStatus
from experiments.pipelines.analysis.pipeline import (
    AnalysisPipelineContext,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)


def save_table_artifact(
    state: AnalysisPipelineState[pl.DataFrame],
    context: AnalysisPipelineContext,
    *,
    artifact_suffix: str = ".tex",
    float_format: str = "%.4f",
) -> AnalysisPipelineTaskResult[pl.DataFrame]:
    """Save a DataFrame as a LaTeX table artifact.

    Converts the DataFrame to a LaTeX string and persists it using the
    analysis artifacts repository.

    Args:
        state: The current state containing the DataFrame in result_data.
        context: The pipeline context with repository access.
        artifact_suffix: File extension for the artifact (default: '.tex').
        float_format: Format string for floating point numbers in LaTeX.

    Returns:
        AnalysisPipelineTaskResult: Updated state with artifact bytes,
        or failure status if result_data is missing.
    """
    result_data = state.get("result_data")

    if result_data is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No result_data available to save as table artifact.",
        )

    # Convert Polars DataFrame to pandas for to_latex() support
    pdf = result_data.to_pandas()

    # Generate LaTeX table string
    latex_str = pdf.to_latex(
        index=False,
        float_format=float_format,
        escape=False,
        caption=f"Results for {context.dataset.value}",
        label=f"tab:{context.analysis_name}",
    )

    # Convert to bytes
    artifact_bytes = latex_str.encode("utf-8")

    # Determine artifact name
    artifact_name = f"{context.analysis_name}{artifact_suffix}"

    # Get locale from translator if available
    locale = context.translator.locale.value if context.translator else "en_US"

    # Save via repository
    context.analysis_artifacts_repository.save_analysis_artifact(
        context.dataset,
        artifact_name,
        artifact_bytes,
        locale=locale,
    )

    # Store artifact in state for reference
    updated_state: AnalysisPipelineState[pl.DataFrame] = {
        **state,
        "artifact": BytesIO(artifact_bytes),
    }

    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Saved table artifact: {artifact_name}",
    )


def save_figure_artifact(
    state: AnalysisPipelineState[plt.Figure],
    context: AnalysisPipelineContext,
    *,
    artifact_suffix: str = ".png",
    dpi: int = 150,
    format: str = "png",
) -> AnalysisPipelineTaskResult[plt.Figure]:
    """Save a matplotlib Figure as a PNG artifact.

    Renders the Figure to PNG bytes and persists it using the
    analysis artifacts repository.

    Args:
        state: The current state containing the Figure in result_data.
        context: The pipeline context with repository access.
        artifact_suffix: File extension for the artifact (default: '.png').
        dpi: Resolution for the saved figure.
        format: Image format (default: 'png').

    Returns:
        AnalysisPipelineTaskResult: Updated state with artifact bytes,
        or failure status if result_data is missing.
    """
    result_data = state.get("result_data")

    if result_data is None:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "No result_data available to save as figure artifact.",
        )

    # Render figure to bytes
    buffer = BytesIO()
    result_data.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    artifact_bytes = buffer.read()

    # Close the figure to free memory
    plt.close(result_data)

    # Determine artifact name
    artifact_name = f"{context.analysis_name}{artifact_suffix}"

    # Get locale from translator if available
    locale = context.translator.locale.value if context.translator else "en_US"

    # Save via repository
    context.analysis_artifacts_repository.save_analysis_artifact(
        context.dataset,
        artifact_name,
        artifact_bytes,
        locale=locale,
    )

    # Store artifact in state for reference
    updated_state: AnalysisPipelineState[plt.Figure] = {
        **state,
        "artifact": BytesIO(artifact_bytes),
    }

    return TaskResult(
        updated_state,
        TaskStatus.SUCCESS,
        f"Saved figure artifact: {artifact_name}",
    )


def create_save_table_task(
    artifact_suffix: str = ".tex",
    float_format: str = "%.4f",
) -> Any:
    """Factory function to create a table saving task with custom options.

    Args:
        artifact_suffix: File extension for the artifact.
        float_format: Format string for floating point numbers.

    Returns:
        A task function configured with the specified options.
    """

    def task(
        state: AnalysisPipelineState[pl.DataFrame],
        context: AnalysisPipelineContext,
    ) -> AnalysisPipelineTaskResult[pl.DataFrame]:
        return save_table_artifact(
            state,
            context,
            artifact_suffix=artifact_suffix,
            float_format=float_format,
        )

    return task


def create_save_figure_task(
    artifact_suffix: str = ".png",
    dpi: int = 150,
    format: str = "png",
) -> Any:
    """Factory function to create a figure saving task with custom options.

    Args:
        artifact_suffix: File extension for the artifact.
        dpi: Resolution for the saved figure.
        format: Image format.

    Returns:
        A task function configured with the specified options.
    """

    def task(
        state: AnalysisPipelineState[plt.Figure],
        context: AnalysisPipelineContext,
    ) -> AnalysisPipelineTaskResult[plt.Figure]:
        return save_figure_artifact(
            state,
            context,
            artifact_suffix=artifact_suffix,
            dpi=dpi,
            format=format,
        )

    return task


# Task aliases for explicit naming in pipelines
SaveTableArtifactTask = save_table_artifact
"""Task to save a DataFrame as a LaTeX table artifact."""

SaveFigureArtifactTask = save_figure_artifact
"""Task to save a matplotlib Figure as a PNG artifact."""


__all__ = [
    "SaveTableArtifactTask",
    "SaveFigureArtifactTask",
    "save_table_artifact",
    "save_figure_artifact",
    "create_save_table_task",
    "create_save_figure_task",
]
