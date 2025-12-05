"""Tests for the analysis exporters module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.core.analysis.constants import METRIC_ACCURACY_BALANCED, METRIC_F1_SCORE
from experiments.core.analysis.exporters import (
    BaseExporter,
    CompositeExporter,
    CsvExporter,
    FigureExporter,
    LatexExporter,
    StabilityFigureExporter,
)
from experiments.core.data import Dataset


def identity_translate(s: str) -> str:
    """Identity translation function for testing."""
    return s


@pytest.fixture
def stability_data() -> dict[str, Any]:
    """Create sample stability analysis data."""
    return {
        "data": pd.DataFrame(
            {
                "model": ["rf", "rf", "svm", "svm"],
                "technique": ["baseline", "smote", "baseline", "smote"],
                "model_display": ["Random Forest", "Random Forest", "SVM", "SVM"],
                "technique_display": ["Baseline", "SMOTE", "Baseline", "SMOTE"],
                "accuracy_balanced": [0.85, 0.87, 0.82, 0.84],
                "f1_score": [0.75, 0.78, 0.72, 0.74],
            }
        ),
        "metrics": [METRIC_ACCURACY_BALANCED, METRIC_F1_SCORE],
        "metric_names": {
            "accuracy_balanced": "Balanced Accuracy",
            "f1_score": "F1 Score",
        },
        "dataset_display": "Taiwan Credit",
        "analysis_type": "stability",
    }


@pytest.fixture
def summary_data() -> dict[str, Any]:
    """Create sample experiment summary data."""
    return {
        "data": pd.DataFrame(
            {
                "model": ["rf", "svm"],
                "technique": ["baseline", "baseline"],
                "model_display": ["Random Forest", "SVM"],
                "technique_display": ["Baseline", "Baseline"],
                "accuracy_balanced_mean": [0.85, 0.82],
                "accuracy_balanced_std": [0.02, 0.03],
            }
        ),
        "display_df": pd.DataFrame(
            {
                "Model": ["Random Forest", "SVM"],
                "Technique": ["Baseline", "Baseline"],
                "Balanced Accuracy": ["0.8500 ± 0.0200", "0.8200 ± 0.0300"],
            }
        ),
        "metrics": [METRIC_ACCURACY_BALANCED],
        "metric_names": {"accuracy_balanced": "Balanced Accuracy"},
        "dataset_display": "Taiwan Credit",
        "analysis_type": "experiment_summary",
    }


class DescribeBaseExporter:
    def it_is_abstract(self):
        with pytest.raises(TypeError):
            BaseExporter(identity_translate)  # type: ignore


class DescribeFigureExporter:
    def it_has_configurable_dpi(self):
        class ConcreteFigureExporter(FigureExporter):
            def export(
                self, data: dict[str, Any], output_dir: Path, dataset: Dataset
            ) -> list[Path]:
                return []

        exporter = ConcreteFigureExporter(identity_translate, dpi=150)
        assert exporter._dpi == 150

    def it_has_configurable_theme_style(self):
        class ConcreteFigureExporter(FigureExporter):
            def export(
                self, data: dict[str, Any], output_dir: Path, dataset: Dataset
            ) -> list[Path]:
                return []

        exporter = ConcreteFigureExporter(identity_translate, theme_style="darkgrid")
        assert exporter._theme_style == "darkgrid"


class DescribeStabilityFigureExporter:
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def it_exports_figures_for_each_metric(
        self,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        tmp_path: Path,
        stability_data: dict[str, Any],
    ):
        exporter = StabilityFigureExporter(identity_translate)

        paths = exporter.export(stability_data, tmp_path, Dataset.TAIWAN_CREDIT)

        # Should create one figure per metric
        assert len(paths) == 2
        assert mock_savefig.call_count == 2

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def it_returns_correct_file_paths(
        self,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        tmp_path: Path,
        stability_data: dict[str, Any],
    ):
        exporter = StabilityFigureExporter(identity_translate)

        paths = exporter.export(stability_data, tmp_path, Dataset.TAIWAN_CREDIT)

        path_names = [p.name for p in paths]
        assert any("accuracy_balanced" in name for name in path_names)
        assert any("f1_score" in name for name in path_names)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def it_skips_metrics_not_in_dataframe(
        self,
        mock_close: MagicMock,
        mock_savefig: MagicMock,
        tmp_path: Path,
    ):
        data = {
            "data": pd.DataFrame(
                {
                    "model_display": ["RF"],
                    "technique_display": ["Baseline"],
                    "accuracy_balanced": [0.85],
                    # f1_score is missing
                }
            ),
            "metrics": [METRIC_ACCURACY_BALANCED, METRIC_F1_SCORE],
            "metric_names": {
                "accuracy_balanced": "Balanced Accuracy",
                "f1_score": "F1 Score",
            },
            "dataset_display": "Taiwan Credit",
            "analysis_type": "stability",
        }

        exporter = StabilityFigureExporter(identity_translate)
        paths = exporter.export(data, tmp_path, Dataset.TAIWAN_CREDIT)

        # Should only create figure for accuracy_balanced
        assert len(paths) == 1


class DescribeCsvExporter:
    def it_exports_dataframe_to_csv(self, tmp_path: Path, summary_data: dict[str, Any]):
        exporter = CsvExporter(identity_translate)

        paths = exporter.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        assert len(paths) == 1
        assert paths[0].suffix == ".csv"
        assert paths[0].exists()

    def it_uses_display_df_when_available(self, tmp_path: Path, summary_data: dict[str, Any]):
        exporter = CsvExporter(identity_translate)

        paths = exporter.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        # Read back and verify it used display_df
        df = pd.read_csv(paths[0])
        assert "Model" in df.columns
        assert "Technique" in df.columns

    def it_falls_back_to_data_when_no_display_df(self, tmp_path: Path):
        # CsvExporter requires display_df, it returns empty when not present
        data = {
            "data": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
            "analysis_type": "test",
        }
        exporter = CsvExporter(identity_translate)

        paths = exporter.export(data, tmp_path, Dataset.TAIWAN_CREDIT)

        # CsvExporter only exports when display_df is present
        assert paths == []


class DescribeLatexExporter:
    def it_exports_dataframe_to_latex(self, tmp_path: Path, summary_data: dict[str, Any]):
        exporter = LatexExporter(identity_translate)

        paths = exporter.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        # LatexExporter creates main file + per-technique files
        assert len(paths) >= 1
        assert paths[0].suffix == ".tex"
        assert paths[0].exists()

    def it_creates_valid_latex_content(self, tmp_path: Path, summary_data: dict[str, Any]):
        exporter = LatexExporter(identity_translate)

        paths = exporter.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        content = paths[0].read_text()
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content


class DescribeCompositeExporter:
    def it_calls_all_child_exporters(self, tmp_path: Path, summary_data: dict[str, Any]):
        mock_exporter1 = MagicMock()
        mock_exporter1.export.return_value = [tmp_path / "file1.csv"]

        mock_exporter2 = MagicMock()
        mock_exporter2.export.return_value = [tmp_path / "file2.tex"]

        composite = CompositeExporter(identity_translate, [mock_exporter1, mock_exporter2])

        paths = composite.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        mock_exporter1.export.assert_called_once()
        mock_exporter2.export.assert_called_once()
        assert len(paths) == 2

    def it_aggregates_paths_from_all_exporters(self, tmp_path: Path, summary_data: dict[str, Any]):
        csv_exporter = CsvExporter(identity_translate)
        latex_exporter = LatexExporter(identity_translate)
        composite = CompositeExporter(identity_translate, [csv_exporter, latex_exporter])

        paths = composite.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        suffixes = [p.suffix for p in paths]
        assert ".csv" in suffixes
        assert ".tex" in suffixes

    def it_handles_empty_exporter_list(self, tmp_path: Path, summary_data: dict[str, Any]):
        composite = CompositeExporter(identity_translate, [])

        paths = composite.export(summary_data, tmp_path, Dataset.TAIWAN_CREDIT)

        assert paths == []
