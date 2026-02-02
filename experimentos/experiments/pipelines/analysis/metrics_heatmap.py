import matplotlib.pyplot as plt

from experiments.pipelines.analysis.base import (
    AnalysisPipeline,
    AnalysisPipelineFactory,
)
from experiments.pipelines.analysis.tasks.common import compute_metrics
from experiments.pipelines.analysis.tasks.generation import generate_metrics_heatmap

type MetricsHeatmap = plt.Figure
"""Represents the metrics heatmap analysis result as a matplotlib Figure."""


class MetricsHeatmapPipelineFactory(AnalysisPipelineFactory[MetricsHeatmap]):
    """Factory for creating metrics heatmap analysis pipelines."""

    _NAME = "MetricsHeatmap"

    def _add_analysis_steps(
        self,
        pipeline: AnalysisPipeline[MetricsHeatmap],
    ) -> None:
        # Step 1: Compute metrics from raw predictions
        pipeline.add_step(
            name="ComputeMetrics",
            task=compute_metrics,
        )

        # Step 2: Generate the metrics heatmap
        pipeline.add_step(
            name="GenerateMetricsHeatmap",
            task=generate_metrics_heatmap,  # type: ignore[arg-type]
        )
