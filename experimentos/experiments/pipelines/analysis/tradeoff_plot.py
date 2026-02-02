import matplotlib.pyplot as plt

from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineFactory,
)
from experiments.pipelines.analysis.tasks.common import compute_metrics
from experiments.pipelines.analysis.tasks.generation import generate_tradeoff_plot

type TradeoffPlot = plt.Figure
"""Represents the tradeoff plot analysis result as a matplotlib Figure."""


class TradeoffPlotPipelineFactory(AnalysisPipelineFactory[TradeoffPlot]):
    """Factory for creating precision-sensitivity tradeoff plot pipelines."""

    _NAME = "TradeoffPlot"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[TradeoffPlot],
    ) -> None:
        # Step 1: Compute metrics from raw predictions
        pipeline.add_step(
            name="ComputeMetrics",
            task=compute_metrics,
        )

        # Step 2: Generate the tradeoff plot
        pipeline.add_step(
            name="GenerateTradeoffPlot",
            task=generate_tradeoff_plot,  # type: ignore[arg-type]
        )
