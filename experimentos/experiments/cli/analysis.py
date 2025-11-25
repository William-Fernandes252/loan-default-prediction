"""CLI for analyzing experimental results."""

import ast
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from typing_extensions import Annotated

from experiments.context import Context
from experiments.core.data import Dataset

MODULE_NAME = "experiments.cli.analysis"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


# Estimated Majority/Minority ratios for the datasets
# Used to populate the 'imbalance_ratio' column for cross-dataset analysis
IMBALANCE_RATIOS = {
    "lending_club": 9.0,  # ~90% vs 10%
    "taiwan_credit": 3.5,  # ~78% vs 22%
    "corporate_credit": 2000.0,  # ~99.95% vs 0.05%
}


def _load_data(ctx: Context, dataset: Dataset) -> pd.DataFrame:
    """Loads the consolidated results for a given dataset."""
    path = ctx.get_latest_consolidated_results_path(dataset.value)
    if path is None or not path.exists():
        ctx.logger.warning(f"No consolidated results found for {dataset.value} at {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _ensure_output_dir(ctx: Context, dataset_name: str) -> Path:
    """Creates and returns the directory for saving analysis plots."""
    # Saves in reports/figures/<dataset>/
    # If analyzing multiple datasets combined, uses 'combined'
    output_dir = ctx.cfg.figures_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@app.command("stability")
def analyze_stability_and_variance(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
):
    """Analyzes the stability and variance of model performance across seeds.

    Generates Boxplots grouped by Technique and Model to visualize:
    1. Performance spread (variance): How much results change with random seeds?
    2. Median performance: Which method is consistently better?
    3. Outliers: Cases where the model failed significantly.
    """
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    # Plot settings
    sns.set_theme(style="whitegrid")
    metrics_to_plot = ["roc_auc", "g_mean", "f1_score"]

    for ds in datasets:
        ctx.logger.info(f"Generating stability analysis for {ds.value}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.value)

        for metric in metrics_to_plot:
            if metric not in df.columns:
                continue

            plt.figure(figsize=(14, 8))

            # Boxplot: X=Technique, Y=Metric, Hue=Model
            # Grouping by Model allows comparing techniques within a model architecture
            ax = sns.boxplot(
                data=df,
                x="technique",
                y=metric,
                hue="model",
                palette="viridis",
                showfliers=False,  # Hide outliers to focus on the IQR boxes
                linewidth=1.5,
            )

            # Add stripplot to see individual points (seeds) distributions
            sns.stripplot(
                data=df,
                x="technique",
                y=metric,
                hue="model",
                dodge=True,
                alpha=0.4,
                palette="dark:black",
                legend=False,
                ax=ax,
                size=3,
            )

            metric_name = metric.replace("_", " ").upper()
            plt.title(f"Stability Analysis: {metric_name} - {ds.value}", fontsize=16)
            plt.ylabel(metric_name, fontsize=12)
            plt.xlabel("Handling Technique", fontsize=12)
            plt.xticks(rotation=15)
            plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Model")

            filename = output_dir / f"stability_{metric}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()

            ctx.logger.success(f"Saved stability plot to {filename}")


@app.command("risktradeoff")
def analyze_risk_tradeoff(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
):
    """Analyzes the risk trade-off curves for different models and techniques.

    In credit risk, a false negative (not identifying a defaulter) is costly.
    However, rejecting many good payers (false positives) results in lost revenue.
    Which technique offered the best Recall without destroying Precision?
    The F1-Score helps summarize this, but the graph tells the full story.
    """
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    for ds in datasets:
        ctx.logger.info(f"Generating risk trade-off analysis for {ds.value}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.value)

        # Calculate mean performance across seeds for stability
        df_agg = df.groupby(["model", "technique"])[["precision", "recall"]].mean().reset_index()

        plt.figure(figsize=(12, 10))

        # Scatter plot:
        # X=Recall (Sensitivity), Y=Precision
        # Hue=Technique (to see the shift caused by resampling/cost)
        # Style=Model (to differentiate architectures)
        sns.scatterplot(
            data=df_agg,
            x="recall",
            y="precision",
            hue="technique",
            style="model",
            s=200,  # Marker size
            palette="deep",
            alpha=0.8,
            edgecolor="k",
        )

        plt.title(
            f"Precision-Recall Trade-off (Mean over {ctx.cfg.num_seeds} seeds) - {ds.value}",
            fontsize=16,
        )
        plt.xlabel("Recall (Sensitivity) - Ability to detect defaults", fontsize=12)
        plt.ylabel("Precision - Trustworthiness of default prediction", fontsize=12)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

        # Add F1 Score isolines for reference
        import numpy as np

        f_scores = np.linspace(0.2, 0.8, num=4)
        for f in f_scores:
            x = np.linspace(0.01, 1)
            y = f * x / (2 * x - f)
            plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2, linestyle="--")
            plt.text(1.0, f / (2 - f), f"f1={f:.1f}", alpha=0.3)

        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)

        filename = output_dir / "risk_tradeoff_scatter.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        ctx.logger.success(f"Saved risk trade-off plot to {filename}")


@app.command("imbalanceimpact")
def analyze_imbalance_impact(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
):
    """Analyzes how class imbalance ratio affects model performance.

    Loads each requested dataset independently, injects its imbalance ratio,
    and generates scatter plots to visualize the correlation between imbalance
    ratio and key performance metrics (ROC AUC, F1 Score).
    """
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    metrics = ["roc_auc", "f1_score"]

    for ds in datasets:
        ctx.logger.info(f"Generating imbalance impact analysis for {ds.value}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        ratio = IMBALANCE_RATIOS.get(ds.value, 1.0)
        df_plot = df.copy()
        df_plot["imbalance_ratio"] = ratio

        available_metrics = [metric for metric in metrics if metric in df_plot.columns]
        if not available_metrics:
            ctx.logger.warning(f"No supported metrics found for imbalance analysis in {ds.value}.")
            continue

        output_dir = _ensure_output_dir(ctx, ds.value)

        plt.figure(figsize=(7 * len(available_metrics), 6))

        for i, metric in enumerate(available_metrics, 1):
            plt.subplot(1, len(available_metrics), i)
            sns.scatterplot(
                data=df_plot,
                x="imbalance_ratio",
                y=metric,
                hue="technique",
                style="model",
                s=100,
                palette="muted",
                alpha=0.7,
                edgecolor="k",
            )
            plt.xscale("log")
            plt.xlabel("Imbalance Ratio (Majority/Minority) - Log Scale", fontsize=12)
            plt.ylabel(metric.replace("_", " ").upper(), fontsize=12)
            plt.title(
                f"{metric.replace('_', ' ').upper()} vs. Imbalance Ratio - {ds.value}",
                fontsize=14,
            )
            plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)

        filename = output_dir / "imbalance_impact.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        ctx.logger.success(f"Saved imbalance impact plot to {filename}")


@app.command("csvsresampling")
def compare_cost_sensitive_and_resampling(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
):
    """Compares cost-sensitive methods against resampling techniques.

    Generates bar plots to visualize and compare the performance of
    cost-sensitive classifiers (e.g., MetaCost) against various resampling methods.
    """
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    for ds in datasets:
        ctx.logger.info(f"Generating cost-sensitive vs resampling analysis for {ds.value}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.value)

        plt.figure(figsize=(12, 8))

        sns.barplot(
            data=df,
            x="technique",
            y="roc_auc",
            hue="model",
            palette="Set2",
            errorbar="sd",
            capsize=0.1,
        )

        plt.title(f"Cost-Sensitive vs Resampling Performance - {ds.value}", fontsize=16)
        plt.ylabel("ROC AUC", fontsize=12)
        plt.xlabel("Technique", fontsize=12)
        plt.ylim(0.0, 1.0)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Model")
        plt.xticks(rotation=15)

        filename = output_dir / "cost_sensitive_vs_resampling.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        ctx.logger.success(f"Saved cost-sensitive vs resampling plot to {filename}")


@app.command("hyperparameters")
def analyze_hyperparameter_effects(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
):
    """Analyzes the effects of hyperparameter choices on model performance.

    Generates heatmaps or line plots to visualize how different hyperparameter
    settings impact key performance metrics like ROC AUC.
    """
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    for ds in datasets:
        ctx.logger.info(f"Generating hyperparameter effects analysis for {ds.value}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.value)

        # Safely parse the 'best_params' string into columns
        # best_params is typically a string representation of a dict: "{'clf__C': 1.0, ...}"
        try:
            hp_df = df["best_params"].apply(lambda x: pd.Series(ast.literal_eval(x)))
        except (ValueError, SyntaxError) as e:
            ctx.logger.error(f"Failed to parse hyperparameters for {ds.value}: {e}")
            continue

        # Combine metrics with extracted hyperparameters
        metric_columns = [
            col
            for col in ("roc_auc", "g_mean", "f1_score", "precision", "recall")
            if col in df.columns
        ]

        # We also need 'technique' and 'model' for grouping
        meta_columns = ["technique", "model"]

        if metric_columns:
            # Reconstruct a DataFrame with HPs + Metrics + Metadata
            merged_df = pd.concat(
                [
                    df[meta_columns + metric_columns].reset_index(drop=True),
                    hp_df.reset_index(drop=True),
                ],
                axis=1,
            )

        # Example: If analyzing 'clf__alpha' (MLP) or 'clf__C' (SVM/LR)
        # You can extend this list based on your grid search space
        target_params = ["clf__alpha", "clf__C", "clf__learning_rate"]

        for param in target_params:
            if param in merged_df.columns:
                plt.figure(figsize=(10, 6))

                # Check if param is numeric to decide on scale
                is_numeric = pd.api.types.is_numeric_dtype(merged_df[param])

                sns.lineplot(
                    data=merged_df,
                    x=param,
                    y="roc_auc",
                    hue="technique",
                    style="model",
                    marker="o",
                    err_style="bars",  # error bars for multiple seeds
                    errorbar=("se", 1),  # Standard Error
                )

                if is_numeric:
                    plt.xscale("log")

                plt.title(f"Effect of {param} on ROC AUC - {ds.value}", fontsize=16)
                plt.xlabel(f"{param} (Log Scale if numeric)", fontsize=12)
                plt.ylabel("ROC AUC", fontsize=12)
                plt.ylim(0.0, 1.05)
                plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Technique")

                filename = output_dir / f"hyperparameter_effects_{param.replace('__', '_')}.png"
                plt.tight_layout()
                plt.savefig(filename, dpi=300)
                plt.close()

                ctx.logger.success(f"Saved hyperparameter effects plot to {filename}")


@app.command("all")
def run_all_analyses(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
):
    """Runs all analysis commands sequentially."""
    analyze_stability_and_variance(dataset)
    analyze_risk_tradeoff(dataset)
    analyze_imbalance_impact(dataset)
    compare_cost_sensitive_and_resampling(dataset)
    analyze_hyperparameter_effects(dataset)


if __name__ == "__main__":
    app()
