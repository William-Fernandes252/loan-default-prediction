from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineFactory,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)

type Stability = dict[str, float]
"""Represents the stability analysis result as a dictionary of metrics."""


def perform_stability_analysis(
    state: AnalysisPipelineState[Stability], context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult[Stability]:
    """Performs stability analysis on the model predictions.

    Args:
        state: The current state of the analysis pipeline.
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult[Stability]: The updated state with stability analysis results.
    """
    raise NotImplementedError("Stability analysis not yet implemented.")


class StabilityAnalysisPipelineFactory(AnalysisPipelineFactory[Stability]):
    _NAME = "StabilityAnalysis"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[dict[str, float]],
    ) -> None:
        pipeline.add_step(
            name="PerformStabilityAnalysis",
            task=perform_stability_analysis,
        )
