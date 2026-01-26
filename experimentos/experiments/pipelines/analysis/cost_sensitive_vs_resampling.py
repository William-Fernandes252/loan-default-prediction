from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineContext,
    AnalysisPipelineFactory,
    AnalysisPipelineState,
    AnalysisPipelineTaskResult,
)

type CostSensitiveVsResamplingComparison = dict[str, float]
"""Represents the cost-sensitive vs resampling comparison result as a dictionary of metrics."""


def perform_cost_sensitive_vs_resampling_comparison(
    state: AnalysisPipelineState[CostSensitiveVsResamplingComparison],
    context: AnalysisPipelineContext,
) -> AnalysisPipelineTaskResult[CostSensitiveVsResamplingComparison]:
    """Performs cost-sensitive vs resampling comparison on the model predictions.

    Args:
        state: The current state of the analysis pipeline.
        context: The context of the analysis pipeline.

    Returns:
        AnalysisPipelineTaskResult[CostSensitiveVsResamplingComparison]: The updated state with cost-sensitive vs resampling comparison results.
    """
    raise NotImplementedError("Cost-sensitive vs resampling comparison not yet implemented.")


class CostSensitiveVsResamplingComparisonPipelineFactory(
    AnalysisPipelineFactory[CostSensitiveVsResamplingComparison]
):
    _NAME = "CostSensitiveVsResamplingComparison"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[dict[str, float]],
    ) -> None:
        pipeline.add_step(
            name="PerformCostSensitiveVsResamplingComparison",
            task=perform_cost_sensitive_vs_resampling_comparison,
        )
