"""Data exporter implementations for the analysis pipeline.

This module provides concrete implementations of the DataExporter protocol
for various output formats (figures, CSV, LaTeX, etc.) and includes a
composite exporter for combining multiple exporters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.core.analysis.metrics import DEFAULT_FIGURE_DPI, DEFAULT_THEME_STYLE
from experiments.core.analysis.protocols import TranslationFunc
from experiments.core.data import Dataset


class BaseExporter(ABC):
    """Base class for data exporters with common functionality."""

    def __init__(self, translate: TranslationFunc) -> None:
        """Initialize the exporter.

        Args:
            translate: Translation function for display names.
        """
        self._translate = translate

    @abstractmethod
    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export the analysis results."""
        ...


class FigureExporter(BaseExporter):
    """Base class for figure exporters using matplotlib/seaborn.

    Provides common figure setup and save functionality.
    """

    def __init__(
        self,
        translate: TranslationFunc,
        dpi: int = DEFAULT_FIGURE_DPI,
        theme_style: str = DEFAULT_THEME_STYLE,
    ) -> None:
        """Initialize the figure exporter.

        Args:
            translate: Translation function.
            dpi: Figure resolution in dots per inch.
            theme_style: Seaborn theme style.
        """
        super().__init__(translate)
        self._dpi = dpi
        self._theme_style = theme_style

    def _setup_theme(self) -> None:
        """Configure seaborn theme."""
        sns.set_theme(style=self._theme_style)

    def _save_figure(self, path: Path) -> None:
        """Save the current figure and close it."""
        plt.tight_layout()
        plt.savefig(path, dpi=self._dpi)
        plt.close()


class StabilityFigureExporter(FigureExporter):
    """Exports stability analysis as boxplot figures."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export stability boxplots for each metric.

        Args:
            data: Transformed data containing 'data', 'metrics', 'metric_names',
                  and 'dataset_display'.
            output_dir: Directory to save figures to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved figures.
        """
        self._setup_theme()
        df = data["data"]
        metrics = data["metrics"]
        metric_names = data["metric_names"]
        ds_display = data["dataset_display"]

        exported_paths: list[Path] = []

        for metric_cfg in metrics:
            metric = metric_cfg.id
            if metric not in df.columns:
                continue

            plt.figure(figsize=(14, 8))

            ax = sns.boxplot(
                data=df,
                x="technique_display",
                y=metric,
                hue="model_display",
                palette="viridis",
                showfliers=False,
                linewidth=1.5,
            )

            sns.stripplot(
                data=df,
                x="technique_display",
                y=metric,
                hue="model_display",
                dodge=True,
                alpha=0.4,
                palette="dark:black",
                legend=False,
                ax=ax,
                size=3,
            )

            metric_display = metric_names.get(metric, metric.replace("_", " ").upper())

            title = self._translate("Stability Analysis: {metric_name} - {dataset_name}").format(
                metric_name=metric_display, dataset_name=ds_display
            )
            plt.title(title, fontsize=16)
            plt.ylabel(metric_display, fontsize=12)
            plt.xlabel(self._translate("Handling Technique"), fontsize=12)
            plt.xticks(rotation=15)
            plt.legend(
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                title=self._translate("Model"),
            )

            filename = output_dir / f"stability_{metric}.png"
            self._save_figure(filename)
            exported_paths.append(filename)

        return exported_paths


class RiskTradeoffFigureExporter(FigureExporter):
    """Exports precision-recall tradeoff scatter plots."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export risk tradeoff scatter plot.

        Args:
            data: Transformed data containing 'data' and 'dataset_display'.
            output_dir: Directory to save figures to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved figures.
        """
        self._setup_theme()
        df_agg = data["data"]
        ds_display = data["dataset_display"]

        plt.figure(figsize=(12, 10))

        sns.scatterplot(
            data=df_agg,
            x="recall",
            y="precision",
            hue="technique_display",
            style="model_display",
            s=200,
            palette="deep",
            alpha=0.8,
            edgecolor="k",
        )

        title = self._translate("Precision-Sensitivity Trade-off - {dataset_name}").format(
            dataset_name=ds_display
        )
        plt.title(title, fontsize=16)
        plt.xlabel(
            self._translate("Recall (Sensitivity) - Ability to detect defaults"),
            fontsize=12,
        )
        plt.ylabel(
            self._translate("Precision - Trustworthiness of default prediction"),
            fontsize=12,
        )
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

        # Add F1 Score isolines for reference
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f in f_scores:
            x = np.linspace(0.01, 1)
            y = f * x / (2 * x - f)
            plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2, linestyle="--")
            plt.text(1.0, f / (2 - f), f"f1={f:.1f}", alpha=0.3)

        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)

        filename = output_dir / "risk_tradeoff_scatter.png"
        self._save_figure(filename)

        return [filename]


class ImbalanceImpactFigureExporter(FigureExporter):
    """Exports imbalance impact scatter plots."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export imbalance impact scatter plots.

        Args:
            data: Transformed data with 'data', 'metrics', 'metric_names'.
            output_dir: Directory to save figures to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved figures.
        """
        self._setup_theme()
        df_plot = data["data"]
        metrics = data["metrics"]
        metric_names = data["metric_names"]
        ds_display = data["dataset_display"]

        available_metrics = [m for m in metrics if m.id in df_plot.columns]
        if not available_metrics:
            return []

        plt.figure(figsize=(7 * len(available_metrics), 6))

        for i, metric_cfg in enumerate(available_metrics, 1):
            metric = metric_cfg.id
            plt.subplot(1, len(available_metrics), i)
            sns.scatterplot(
                data=df_plot,
                x="imbalance_ratio",
                y=metric,
                hue="technique_display",
                style="model_display",
                s=100,
                palette="muted",
                alpha=0.7,
                edgecolor="k",
            )
            plt.xscale("log")
            plt.xlabel(
                self._translate("Imbalance Ratio (Majority/Minority) - Log Scale"),
                fontsize=12,
            )

            metric_display = metric_names.get(metric, metric.replace("_", " ").upper())
            plt.ylabel(metric_display, fontsize=12)

            title = self._translate("{metric_name} vs. Imbalance Ratio - {dataset_name}").format(
                metric_name=metric_display, dataset_name=ds_display
            )
            plt.title(title, fontsize=14)
            plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)

        filename = output_dir / "imbalance_impact.png"
        self._save_figure(filename)

        return [filename]


class CostSensitiveVsResamplingFigureExporter(FigureExporter):
    """Exports cost-sensitive vs resampling comparison bar plots."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export comparison bar plot.

        Args:
            data: Transformed data containing 'data' and 'dataset_display'.
            output_dir: Directory to save figures to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved figures.
        """
        self._setup_theme()
        df = data["data"]
        ds_display = data["dataset_display"]

        plt.figure(figsize=(12, 8))

        sns.barplot(
            data=df,
            x="technique_display",
            y="accuracy_balanced",
            hue="model_display",
            palette="Set2",
            errorbar="sd",
            capsize=0.1,
        )

        title = self._translate(
            "Cost-Sensitive vs Resampling Performance - {dataset_name}"
        ).format(dataset_name=ds_display)
        plt.title(title, fontsize=16)
        plt.ylabel(self._translate("Balanced Accuracy"), fontsize=12)
        plt.xlabel(self._translate("Technique"), fontsize=12)
        plt.ylim(0.0, 1.0)
        plt.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            title=self._translate("Model"),
        )
        plt.xticks(rotation=15)

        filename = output_dir / "cost_sensitive_vs_resampling.png"
        self._save_figure(filename)

        return [filename]


class HyperparameterFigureExporter(FigureExporter):
    """Exports hyperparameter effects line plots."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export hyperparameter effect plots.

        Args:
            data: Transformed data with 'data', 'target_params', 'parse_error'.
            output_dir: Directory to save figures to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved figures.
        """
        if data.get("parse_error"):
            return []

        self._setup_theme()
        merged_df = data["data"]
        target_params = data["target_params"]
        ds_display = data["dataset_display"]

        exported_paths: list[Path] = []

        for param in target_params:
            if param not in merged_df.columns:
                continue

            plt.figure(figsize=(10, 6))

            is_numeric = pd.api.types.is_numeric_dtype(merged_df[param])

            sns.lineplot(
                data=merged_df,
                x=param,
                y="accuracy_balanced",
                hue="technique_display"
                if "technique_display" in merged_df.columns
                else "technique",
                style="model_display" if "model_display" in merged_df.columns else "model",
                marker="o",
                errorbar=None,
            )

            if is_numeric:
                plt.xscale("log")

            title = self._translate(
                "Effect of {param} on Balanced Accuracy - {dataset_name}"
            ).format(param=param, dataset_name=ds_display)
            plt.title(title, fontsize=16)

            xlabel = self._translate("{param} (Log Scale if numeric)").format(param=param)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(self._translate("Balanced Accuracy"), fontsize=12)
            plt.ylim(0.0, 1.05)
            plt.legend(
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                title=self._translate("Technique"),
            )

            filename = output_dir / f"hyperparameter_effects_{param.replace('__', '_')}.png"
            self._save_figure(filename)
            exported_paths.append(filename)

        return exported_paths


class ExperimentSummaryFigureExporter(FigureExporter):
    """Exports experiment summary as table images."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export summary tables as PNG images per technique.

        Args:
            data: Transformed data with 'display_df'.
            output_dir: Directory to save figures to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved figures.
        """
        display_df = data.get("display_df")
        if display_df is None or display_df.empty:
            return []

        self._setup_theme()
        exported_paths: list[Path] = []

        technique_col = self._translate("Technique")
        techniques = display_df[technique_col].unique()

        for tech in techniques:
            sub_df: pd.DataFrame = display_df[display_df[technique_col] == tech]

            row_height = 0.5
            header_height = 0.8
            fig_height = header_height + (len(sub_df) * row_height)

            plt.figure(figsize=(14, fig_height))
            plt.axis("off")

            sub_df = sub_df.drop(columns=[technique_col])
            table = plt.table(
                cellText=sub_df.values,
                colLabels=sub_df.columns,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)

            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#f0f0f0")

            safe_tech_name = str(tech).replace(" ", "_").replace("/", "-").lower()
            filename = output_dir / f"results_summary_{safe_tech_name}.png"

            plt.savefig(filename, bbox_inches="tight", dpi=self._dpi)
            plt.close()
            exported_paths.append(filename)

        return exported_paths


class CsvExporter(BaseExporter):
    """Exports data as CSV files."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export data to CSV.

        Args:
            data: Transformed data containing 'display_df'.
            output_dir: Directory to save files to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved files.
        """
        display_df = data.get("display_df")
        if display_df is None or display_df.empty:
            return []

        filename = output_dir / "results_summary.csv"
        display_df.to_csv(filename, index=False)

        return [filename]


class LatexExporter(BaseExporter):
    """Exports data as LaTeX tables."""

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export data to LaTeX.

        Args:
            data: Transformed data containing 'display_df'.
            output_dir: Directory to save files to.
            dataset: The dataset being analyzed.

        Returns:
            List of paths to the saved files.
        """
        display_df = data.get("display_df")
        if display_df is None or display_df.empty:
            return []

        exported_paths: list[Path] = []

        # Main summary table
        display_df_latex = display_df.replace("±", r"$\\pm$", regex=True)
        filename = output_dir / "results_summary.tex"
        display_df_latex.to_latex(
            filename,
            index=False,
            escape=False,
            column_format="ll" + "c" * (len(display_df_latex.columns) - 2),
        )
        exported_paths.append(filename)

        # Per-technique tables
        technique_col = self._translate("Technique")
        if technique_col in display_df.columns:
            techniques = display_df[technique_col].unique()

            for tech in techniques:
                sub_df: pd.DataFrame = display_df[display_df[technique_col] == tech]
                sub_df = sub_df.drop(columns=[technique_col])
                sub_df_latex = sub_df.replace("±", r"$\\pm$", regex=True)

                safe_tech_name = str(tech).replace(" ", "_").replace("/", "-").lower()
                tech_filename = output_dir / f"results_summary_{safe_tech_name}.tex"
                sub_df_latex.to_latex(
                    tech_filename,
                    index=False,
                    escape=False,
                    column_format="l" + "c" * (len(sub_df_latex.columns) - 1),
                )
                exported_paths.append(tech_filename)

        return exported_paths


class CompositeExporter(BaseExporter):
    """Combines multiple exporters into a single exporter.

    This exporter delegates to a list of child exporters, collecting
    all exported paths from each.
    """

    def __init__(
        self,
        translate: TranslationFunc,
        exporters: list[BaseExporter],
    ) -> None:
        """Initialize the composite exporter.

        Args:
            translate: Translation function.
            exporters: List of child exporters to delegate to.
        """
        super().__init__(translate)
        self._exporters = exporters

    def export(
        self,
        data: dict[str, Any],
        output_dir: Path,
        dataset: Dataset,
    ) -> list[Path]:
        """Export using all child exporters.

        Args:
            data: Transformed data.
            output_dir: Directory to save files to.
            dataset: The dataset being analyzed.

        Returns:
            Combined list of paths from all child exporters.
        """
        all_paths: list[Path] = []

        for exporter in self._exporters:
            paths = exporter.export(data, output_dir, dataset)
            all_paths.extend(paths)

        return all_paths


# Re-export protocol for convenience
DataExporter = BaseExporter


__all__ = [
    "DataExporter",
    "BaseExporter",
    "FigureExporter",
    "StabilityFigureExporter",
    "RiskTradeoffFigureExporter",
    "ImbalanceImpactFigureExporter",
    "CostSensitiveVsResamplingFigureExporter",
    "HyperparameterFigureExporter",
    "ExperimentSummaryFigureExporter",
    "CsvExporter",
    "LatexExporter",
    "CompositeExporter",
]
