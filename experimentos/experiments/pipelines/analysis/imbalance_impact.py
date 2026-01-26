from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineFactory,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)

type ImbalanceImpact = dict[str, float]
"""Represents the imbalance impact analysis result as a dictionary of metrics."""


def perform_imbalance_impact_analysis(
    state: AnalysisPipelineState[ImbalanceImpact], context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult[ImbalanceImpact]:
    """Performs imbalance impact analysis on the model predictions.

    Args:
        state: The current state of the analysis pipeline.
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult[ImbalanceImpact]: The updated state with imbalance impact analysis results.
    """
    raise NotImplementedError("Imbalance impact analysis not yet implemented.")


class ImbalanceImpactAnalysisPipelineFactory(AnalysisPipelineFactory[ImbalanceImpact]):
    _NAME = "ImbalanceImpactAnalysis"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[dict[str, float]],
    ) -> None:
        pipeline.add_step(
            name="PerformImbalanceImpactAnalysis",
            task=perform_imbalance_impact_analysis,
        )
