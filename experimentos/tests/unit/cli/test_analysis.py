"""Tests for analysis CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from experiments.core.analysis import Locale
from experiments.core.analysis.metrics import Metric
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import Technique
from experiments.services.predictions_analyzer import AnalysisParams, AnalysisResult, AnalysisType


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_analyzer() -> MagicMock:
    """Fixture providing a mock PredictionsAnalyzer."""
    analyzer = MagicMock()
    analyzer.run_analysis.return_value = AnalysisResult(
        dataset=Dataset.TAIWAN_CREDIT,
        analysis_type=AnalysisType.SUMMARY_TABLE,
        success=True,
    )
    return analyzer


@pytest.fixture
def mock_container(mock_analyzer: MagicMock):
    """Fixture that patches the container to return our mock analyzer."""
    with patch("experiments.cli.analysis.container") as container:
        container.predictions_analyzer.return_value = mock_analyzer
        yield container


@pytest.fixture
def analysis_app(mock_container: MagicMock):
    """Import the app after mocking the container."""
    from experiments.cli.analysis import app

    return app


# ============================================================================
# Tests for helper functions
# ============================================================================


class DescribeResolveDatasets:
    """Tests for the _resolve_datasets helper function."""

    def it_returns_single_dataset_in_list(self) -> None:
        from experiments.cli.analysis import _resolve_datasets

        result = _resolve_datasets(Dataset.TAIWAN_CREDIT)

        assert result == [Dataset.TAIWAN_CREDIT]

    def it_returns_all_datasets_when_none(self) -> None:
        from experiments.cli.analysis import _resolve_datasets

        result = _resolve_datasets(None)

        assert result == list(Dataset)


# ============================================================================
# Tests for summary command
# ============================================================================


class DescribeSummaryCommand:
    """Tests for the `analyze summary` command."""

    def it_generates_summary_table(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["summary", "taiwan_credit"])

        assert result.exit_code == 0
        mock_analyzer.run_analysis.assert_called_once()
        call_args = mock_analyzer.run_analysis.call_args
        assert call_args.kwargs["analysis_type"] == AnalysisType.SUMMARY_TABLE
        params: AnalysisParams = call_args.kwargs["params"]
        assert params.dataset == Dataset.TAIWAN_CREDIT

    def it_filters_by_technique(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["summary", "taiwan_credit", "--technique", "smote"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.technique_filter == Technique.SMOTE

    def it_uses_short_technique_option(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(
            analysis_app, ["summary", "taiwan_credit", "-t", "random_under_sampling"]
        )

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.technique_filter == Technique.RANDOM_UNDER_SAMPLING

    def it_passes_execution_id(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["summary", "taiwan_credit", "-e", "exec-123"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.execution_id == "exec-123"

    def it_passes_force_flag(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["summary", "taiwan_credit", "--force"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.force_overwrite is True

    def it_passes_locale(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["summary", "taiwan_credit", "-l", "en_US"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.locale == Locale.EN_US


# ============================================================================
# Tests for tradeoff command
# ============================================================================


class DescribeTradeoffCommand:
    """Tests for the `analyze tradeoff` command."""

    def it_generates_tradeoff_plot(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["tradeoff", "taiwan_credit"])

        assert result.exit_code == 0
        call_args = mock_analyzer.run_analysis.call_args
        assert call_args.kwargs["analysis_type"] == AnalysisType.TRADEOFF_PLOT
        params: AnalysisParams = call_args.kwargs["params"]
        assert params.dataset == Dataset.TAIWAN_CREDIT

    def it_passes_force_flag(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["tradeoff", "taiwan_credit", "-f"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.force_overwrite is True


# ============================================================================
# Tests for stability command
# ============================================================================


class DescribeStabilityCommand:
    """Tests for the `analyze stability` command."""

    def it_generates_stability_plot(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["stability", "taiwan_credit"])

        assert result.exit_code == 0
        call_args = mock_analyzer.run_analysis.call_args
        assert call_args.kwargs["analysis_type"] == AnalysisType.STABILITY_PLOT
        params: AnalysisParams = call_args.kwargs["params"]
        assert params.dataset == Dataset.TAIWAN_CREDIT
        assert params.metric == Metric.BALANCED_ACCURACY  # default

    def it_accepts_custom_metric(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["stability", "taiwan_credit", "-m", "g_mean"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.metric == Metric.G_MEAN


# ============================================================================
# Tests for imbalance command
# ============================================================================


class DescribeImbalanceCommand:
    """Tests for the `analyze imbalance` command."""

    def it_generates_imbalance_plot(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["imbalance", "taiwan_credit"])

        assert result.exit_code == 0
        call_args = mock_analyzer.run_analysis.call_args
        assert call_args.kwargs["analysis_type"] == AnalysisType.IMBALANCE_IMPACT_PLOT
        params: AnalysisParams = call_args.kwargs["params"]
        assert params.dataset == Dataset.TAIWAN_CREDIT


# ============================================================================
# Tests for comparison command
# ============================================================================


class DescribeComparisonCommand:
    """Tests for the `analyze comparison` command."""

    def it_generates_comparison_plot(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["comparison", "taiwan_credit"])

        assert result.exit_code == 0
        call_args = mock_analyzer.run_analysis.call_args
        assert call_args.kwargs["analysis_type"] == AnalysisType.CS_VS_RESAMPLING_PLOT
        params: AnalysisParams = call_args.kwargs["params"]
        assert params.dataset == Dataset.TAIWAN_CREDIT


# ============================================================================
# Tests for heatmap command
# ============================================================================


class DescribeHeatmapCommand:
    """Tests for the `analyze heatmap` command."""

    def it_generates_heatmap(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["heatmap", "taiwan_credit"])

        assert result.exit_code == 0
        call_args = mock_analyzer.run_analysis.call_args
        assert call_args.kwargs["analysis_type"] == AnalysisType.METRICS_HEATMAP
        params: AnalysisParams = call_args.kwargs["params"]
        assert params.dataset == Dataset.TAIWAN_CREDIT

    def it_passes_gpu_flag(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["heatmap", "taiwan_credit", "--gpu"])

        assert result.exit_code == 0
        params: AnalysisParams = mock_analyzer.run_analysis.call_args.kwargs["params"]
        assert params.use_gpu is True


# ============================================================================
# Tests for all command
# ============================================================================


class DescribeAllCommand:
    """Tests for the `analyze all` command."""

    def it_runs_all_analyses(
        self,
        runner: CliRunner,
        analysis_app,
        mock_analyzer: MagicMock,
    ) -> None:
        result = runner.invoke(analysis_app, ["all", "taiwan_credit"])

        assert result.exit_code == 0
        # Should call run_analysis 6 times (one for each analysis type)
        assert mock_analyzer.run_analysis.call_count == 6

        # Verify all analysis types were called
        analysis_types_called = [
            call.kwargs["analysis_type"] for call in mock_analyzer.run_analysis.call_args_list
        ]
        assert AnalysisType.SUMMARY_TABLE in analysis_types_called
        assert AnalysisType.TRADEOFF_PLOT in analysis_types_called
        assert AnalysisType.STABILITY_PLOT in analysis_types_called
        assert AnalysisType.IMBALANCE_IMPACT_PLOT in analysis_types_called
        assert AnalysisType.CS_VS_RESAMPLING_PLOT in analysis_types_called
        assert AnalysisType.METRICS_HEATMAP in analysis_types_called
