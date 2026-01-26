from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineFactory,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)

type RiskTradeoff = dict[str, float]
"""Represents the risk tradeoff analysis result as a dictionary of metrics."""


def perform_risk_tradeoff_analysis(
    state: AnalysisPipelineState[RiskTradeoff], context: AnalysisPipelineContext
) -> AnalysisPipelineTaskResult[RiskTradeoff]:
    """Performs risk tradeoff analysis on the model predictions.

    Args:
        state: The current state of the analysis pipeline.
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult[RiskTradeoff]: The updated state with risk tradeoff analysis results.
    """
    raise NotImplementedError("Risk tradeoff analysis not yet implemented.")


class RiskTradeoffAnalysisPipelineFactory(AnalysisPipelineFactory[RiskTradeoff]):
    _NAME = "RiskTradeoffAnalysis"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[dict[str, float]],
    ) -> None:
        pipeline.add_step(
            name="PerformRiskTradeoffAnalysis",
            task=perform_risk_tradeoff_analysis,
        )
