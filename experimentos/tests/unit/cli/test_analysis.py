"""Tests for analysis CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from experiments.cli.analysis import (
    AnalysisType,
    _resolve_datasets,
)
from experiments.core.analysis.metrics import Metric
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import Technique


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_storage(mock_container: MagicMock) -> MagicMock:
    """Fixture providing a mock storage from the container."""
    storage = MagicMock()
    mock_container._storage.return_value = storage
    return storage


@pytest.fixture
def analysis_app(mock_storage: MagicMock):
    """Import the app after mocking the container."""
    from experiments.cli.analysis import app

    return app


# ============================================================================
# Tests for helper functions
# ============================================================================


class DescribeResolveDatasets:
    """Tests for the _resolve_datasets helper function."""

    def it_returns_single_dataset_in_list(self) -> None:
        result = _resolve_datasets(Dataset.TAIWAN_CREDIT)

        assert result == [Dataset.TAIWAN_CREDIT]

    def it_returns_all_datasets_when_none(self) -> None:
        result = _resolve_datasets(None)

        assert result == list(Dataset)


class DescribeAnalysisType:
    """Tests for the AnalysisType enum."""

    def it_has_summary_table_type(self) -> None:
        assert AnalysisType.SUMMARY_TABLE == "summary_table"

    def it_has_tradeoff_plot_type(self) -> None:
        assert AnalysisType.TRADEOFF_PLOT == "tradeoff_plot"

    def it_has_stability_plot_type(self) -> None:
        assert AnalysisType.STABILITY_PLOT == "stability_plot"

    def it_has_imbalance_impact_plot_type(self) -> None:
        assert AnalysisType.IMBALANCE_IMPACT_PLOT == "imbalance_impact_plot"

    def it_has_cs_vs_resampling_plot_type(self) -> None:
        assert AnalysisType.CS_VS_RESAMPLING_PLOT == "cs_vs_resampling_plot"

    def it_has_metrics_heatmap_type(self) -> None:
        assert AnalysisType.METRICS_HEATMAP == "metrics_heatmap"


# ============================================================================
# Tests for summary command
# ============================================================================


class DescribeSummaryCommand:
    """Tests for the `analyze summary` command."""

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_summary_table_pipeline")
    def it_generates_summary_table(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["summary", "taiwan_credit"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_once_with(technique=None)

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_summary_table_pipeline")
    def it_filters_by_technique(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["summary", "--technique", "smote"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_with(technique=Technique.SMOTE)

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_summary_table_pipeline")
    def it_uses_short_technique_option(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["summary", "-t", "random_under_sampling"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_with(technique=Technique.RANDOM_UNDER_SAMPLING)

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_summary_table_pipeline")
    def it_passes_execution_id_to_context(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["summary", "taiwan_credit", "-e", "exec-123"])

        assert result.exit_code == 0
        # Verify execution_id was passed to the context
        context = mock_executor.execute.call_args.kwargs["context"]
        assert context.execution_id == "exec-123"

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_summary_table_pipeline")
    def it_uses_long_execution_id_option(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(
            analysis_app, ["summary", "taiwan_credit", "--execution-id", "my-exec-456"]
        )

        assert result.exit_code == 0
        context = mock_executor.execute.call_args.kwargs["context"]
        assert context.execution_id == "my-exec-456"

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_summary_table_pipeline")
    def it_defaults_to_none_execution_id(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["summary", "taiwan_credit"])

        assert result.exit_code == 0
        context = mock_executor.execute.call_args.kwargs["context"]
        assert context.execution_id is None


# ============================================================================
# Tests for tradeoff command
# ============================================================================


class DescribeTradeoffCommand:
    """Tests for the `analyze tradeoff` command."""

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_tradeoff_plot_pipeline")
    def it_generates_tradeoff_plot(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["tradeoff", "taiwan_credit"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_once()

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_tradeoff_plot_pipeline")
    def it_passes_force_flag(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["tradeoff", "--force"])

        assert result.exit_code == 0


# ============================================================================
# Tests for stability command
# ============================================================================


class DescribeStabilityCommand:
    """Tests for the `analyze stability` command."""

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_stability_pipeline")
    def it_generates_stability_plot(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["stability", "taiwan_credit"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_once_with(metric=Metric.BALANCED_ACCURACY)

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_stability_pipeline")
    def it_accepts_custom_metric(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["stability", "--metric", "g_mean"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_with(metric=Metric.G_MEAN)

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_stability_pipeline")
    def it_uses_short_metric_option(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["stability", "-m", "f1_score"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_with(metric=Metric.F1_SCORE)


# ============================================================================
# Tests for imbalance command
# ============================================================================


class DescribeImbalanceCommand:
    """Tests for the `analyze imbalance` command."""

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_imbalance_impact_pipeline")
    def it_generates_imbalance_plot(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["imbalance", "taiwan_credit"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_once_with(metric=Metric.BALANCED_ACCURACY)


# ============================================================================
# Tests for comparison command
# ============================================================================


class DescribeComparisonCommand:
    """Tests for the `analyze comparison` command."""

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_cs_vs_resampling_pipeline")
    def it_generates_comparison_plot(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["comparison", "taiwan_credit"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_once()


# ============================================================================
# Tests for heatmap command
# ============================================================================


class DescribeHeatmapCommand:
    """Tests for the `analyze heatmap` command."""

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_metrics_heatmap_pipeline")
    def it_generates_heatmap(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["heatmap", "taiwan_credit"])

        assert result.exit_code == 0
        mock_build_pipeline.assert_called_once()

    @patch("experiments.cli.analysis.PipelineExecutor")
    @patch("experiments.cli.analysis.build_metrics_heatmap_pipeline")
    def it_passes_gpu_flag(
        self,
        mock_build_pipeline: MagicMock,
        mock_executor_class: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        mock_pipeline = MagicMock()
        mock_build_pipeline.return_value = mock_pipeline
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded.return_value = True
        mock_executor.execute.return_value = mock_result
        mock_executor_class.return_value = mock_executor

        result = runner.invoke(analysis_app, ["heatmap", "--gpu"])

        assert result.exit_code == 0


# ============================================================================
# Tests for all command
# ============================================================================


class DescribeAllCommand:
    """Tests for the `analyze all` command."""

    @patch("experiments.cli.analysis.generate_metrics_heatmap")
    @patch("experiments.cli.analysis.generate_cs_vs_resampling_plot")
    @patch("experiments.cli.analysis.generate_imbalance_impact_plot")
    @patch("experiments.cli.analysis.generate_stability_plot")
    @patch("experiments.cli.analysis.generate_tradeoff_plot")
    @patch("experiments.cli.analysis.generate_summary_table")
    def it_runs_all_analyses(
        self,
        mock_summary: MagicMock,
        mock_tradeoff: MagicMock,
        mock_stability: MagicMock,
        mock_imbalance: MagicMock,
        mock_comparison: MagicMock,
        mock_heatmap: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        result = runner.invoke(analysis_app, ["all", "taiwan_credit"])

        assert result.exit_code == 0
        mock_summary.assert_called_once()
        mock_tradeoff.assert_called_once()
        mock_stability.assert_called_once()
        mock_imbalance.assert_called_once()
        mock_comparison.assert_called_once()
        mock_heatmap.assert_called_once()

    @patch("experiments.cli.analysis.generate_metrics_heatmap")
    @patch("experiments.cli.analysis.generate_cs_vs_resampling_plot")
    @patch("experiments.cli.analysis.generate_imbalance_impact_plot")
    @patch("experiments.cli.analysis.generate_stability_plot")
    @patch("experiments.cli.analysis.generate_tradeoff_plot")
    @patch("experiments.cli.analysis.generate_summary_table")
    def it_passes_options_to_all_analyses(
        self,
        mock_summary: MagicMock,
        mock_tradeoff: MagicMock,
        mock_stability: MagicMock,
        mock_imbalance: MagicMock,
        mock_comparison: MagicMock,
        mock_heatmap: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        result = runner.invoke(analysis_app, ["all", "taiwan_credit", "--force", "--gpu"])

        assert result.exit_code == 0
        # Check that force and gpu flags were passed to each analysis
        for mock_fn in [
            mock_summary,
            mock_tradeoff,
            mock_stability,
            mock_imbalance,
            mock_comparison,
            mock_heatmap,
        ]:
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs["force"] is True
            assert call_kwargs["gpu"] is True

    @patch("experiments.cli.analysis.generate_metrics_heatmap")
    @patch("experiments.cli.analysis.generate_cs_vs_resampling_plot")
    @patch("experiments.cli.analysis.generate_imbalance_impact_plot")
    @patch("experiments.cli.analysis.generate_stability_plot")
    @patch("experiments.cli.analysis.generate_tradeoff_plot")
    @patch("experiments.cli.analysis.generate_summary_table")
    def it_passes_execution_id_to_all_analyses(
        self,
        mock_summary: MagicMock,
        mock_tradeoff: MagicMock,
        mock_stability: MagicMock,
        mock_imbalance: MagicMock,
        mock_comparison: MagicMock,
        mock_heatmap: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        result = runner.invoke(analysis_app, ["all", "taiwan_credit", "-e", "my-execution-id"])

        assert result.exit_code == 0
        # Check that execution_id was passed to each analysis
        for mock_fn in [
            mock_summary,
            mock_tradeoff,
            mock_stability,
            mock_imbalance,
            mock_comparison,
            mock_heatmap,
        ]:
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs["execution_id"] == "my-execution-id"

    @patch("experiments.cli.analysis.generate_metrics_heatmap")
    @patch("experiments.cli.analysis.generate_cs_vs_resampling_plot")
    @patch("experiments.cli.analysis.generate_imbalance_impact_plot")
    @patch("experiments.cli.analysis.generate_stability_plot")
    @patch("experiments.cli.analysis.generate_tradeoff_plot")
    @patch("experiments.cli.analysis.generate_summary_table")
    def it_defaults_execution_id_to_none(
        self,
        mock_summary: MagicMock,
        mock_tradeoff: MagicMock,
        mock_stability: MagicMock,
        mock_imbalance: MagicMock,
        mock_comparison: MagicMock,
        mock_heatmap: MagicMock,
        runner: CliRunner,
        analysis_app,
    ) -> None:
        result = runner.invoke(analysis_app, ["all", "taiwan_credit"])

        assert result.exit_code == 0
        # Check that execution_id defaults to None
        for mock_fn in [
            mock_summary,
            mock_tradeoff,
            mock_stability,
            mock_imbalance,
            mock_comparison,
            mock_heatmap,
        ]:
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs["execution_id"] is None
