"""CLI for analyzing experimental results."""

import ast
import enum
import gettext
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


class _Language(enum.Enum):
    """Supported languages for analysis reports."""

    ENGLISH = "en_US"
    PORTUGUESE_BRAZIL = "pt_BR"


_LanguageOption = Annotated[_Language, typer.Option("--language", "-l", help="Language code")]

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])


app = typer.Typer()


def _noop(s):
    return s


# Global translation function, will be set in main/commands
_ = _noop

# Estimated Majority/Minority ratios for the datasets
# Used to populate the 'imbalance_ratio' column for cross-dataset analysis
IMBALANCE_RATIOS = {
    Dataset.LENDING_CLUB.id: 9.0,  # ~90% vs 10%
    Dataset.TAIWAN_CREDIT.id: 3.5,  # ~78% vs 22%
    Dataset.CORPORATE_CREDIT_RATING.id: 2000.0,  # ~99.95% vs 0.05%
}


def _setup_i18n(language: _Language):
    """Sets up the gettext translation based on the language code."""
    global _
    locales_dir = Path(__file__).parents[2] / "locales"
    try:
        lang = gettext.translation("base", localedir=locales_dir, languages=[language.value])
        lang.install()
        _ = lang.gettext
    except FileNotFoundError:
        # Fallback to null translation (English/Source)
        _ = _noop


def _get_model_display(model_id: str) -> str:
    """Returns the translated display name for a model."""
    # Map IDs to translatable strings
    mapping = {
        "random_forest": _("Random Forest"),
        "svm": _("Support Vector Machine"),
        "xgboost": _("XGBoost"),
        "mlp": _("Multi-Layer Perceptron"),
    }
    return mapping.get(model_id, model_id)


def _get_technique_display(technique_id: str) -> str:
    """Returns the translated display name for a technique."""
    mapping = {
        "baseline": _("Baseline"),
        "smote": _("SMOTE"),
        "random_under_sampling": _("Random Under Sampling"),
        "smote_tomek": _("SMOTE Tomek"),
        "meta_cost": _("Meta Cost"),
        "cs_svm": _("Cost-sensitive SVM"),
    }
    return mapping.get(technique_id, technique_id)


def _get_dataset_display(dataset_id: str) -> str:
    """Returns the translated display name for a dataset."""
    mapping = {
        "corporate_credit_rating": _("Corporate Credit Rating"),
        "lending_club": _("Lending Club"),
        "taiwan_credit": _("Taiwan Credit"),
    }
    return mapping.get(dataset_id, dataset_id)


def _load_data(ctx: Context, dataset: Dataset) -> pd.DataFrame:
    """Loads the consolidated results for a given dataset."""
    path = ctx.get_latest_consolidated_results_path(dataset.id)
    if path is None or not path.exists():
        ctx.logger.warning(f"No consolidated results found for {dataset.display_name} at {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()

    # Apply translations to model and technique columns
    if "model" in df.columns:
        df["model_display"] = df["model"].apply(_get_model_display)
    if "technique" in df.columns:
        df["technique_display"] = df["technique"].apply(_get_technique_display)

    return df


def _ensure_output_dir(ctx: Context, dataset_name: str, language: _Language) -> Path:
    """Creates and returns the directory for saving analysis plots."""
    # Saves in reports/<language>/figures/<dataset>/
    # If analyzing multiple datasets combined, uses 'combined'
    output_dir = ctx.cfg.figures_dir.parent / language.value / "figures" / dataset_name
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
    language: _LanguageOption = _Language.ENGLISH,
):
    """Analyzes the stability and variance of model performance across seeds.

    Generates Boxplots grouped by Technique and Model to visualize:
    1. Performance spread (variance): How much results change with random seeds?
    2. Median performance: Which method is consistently better?
    3. Outliers: Cases where the model failed significantly.
    """
    _setup_i18n(language)
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    # Plot settings
    sns.set_theme(style="whitegrid")
    metrics_to_plot = ["roc_auc", "g_mean", "f1_score"]

    # Metric display names mapping
    metric_names = {
        "roc_auc": _("ROC AUC"),
        "g_mean": _("G-Mean"),
        "f1_score": _("F1 Score"),
    }

    for ds in datasets:
        ds_display = _get_dataset_display(ds.id)
        ctx.logger.info(f"Generating stability analysis for {ds_display}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.id, language)

        for metric in metrics_to_plot:
            if metric not in df.columns:
                continue

            plt.figure(figsize=(14, 8))

            # Boxplot: X=Technique, Y=Metric, Hue=Model
            # Grouping by Model allows comparing techniques within a model architecture
            ax = sns.boxplot(
                data=df,
                x="technique_display",
                y=metric,
                hue="model_display",
                palette="viridis",
                showfliers=False,  # Hide outliers to focus on the IQR boxes
                linewidth=1.5,
            )

            # Add stripplot to see individual points (seeds) distributions
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

            title = _("Stability Analysis: {metric_name} - {dataset_name}").format(
                metric_name=metric_display, dataset_name=ds_display
            )
            plt.title(title, fontsize=16)
            plt.ylabel(metric_display, fontsize=12)
            plt.xlabel(_("Handling Technique"), fontsize=12)
            plt.xticks(rotation=15)
            plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title=_("Model"))

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
    language: _LanguageOption = _Language.ENGLISH,
):
    """Analyzes the risk trade-off curves for different models and techniques.

    In credit risk, a false negative (not identifying a defaulter) is costly.
    However, rejecting many good payers (false positives) results in lost revenue.
    Which technique offered the best Recall without destroying Precision?
    The F1-Score helps summarize this, but the graph tells the full story.
    """
    _setup_i18n(language)
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    for ds in datasets:
        ds_display = _get_dataset_display(ds.id)
        ctx.logger.info(f"Generating risk trade-off analysis for {ds_display}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.id, language)

        # Calculate mean performance across seeds for stability
        df_agg = df.groupby(["model", "technique"])[["precision", "recall"]].mean().reset_index()

        # Apply translations
        df_agg["model_display"] = df_agg["model"].apply(_get_model_display)
        df_agg["technique_display"] = df_agg["technique"].apply(_get_technique_display)

        plt.figure(figsize=(12, 10))

        # Scatter plot:
        # X=Recall (Sensitivity), Y=Precision
        # Hue=Technique (to see the shift caused by resampling/cost)
        # Style=Model (to differentiate architectures)
        sns.scatterplot(
            data=df_agg,
            x="recall",
            y="precision",
            hue="technique_display",
            style="model_display",
            s=200,  # Marker size
            palette="deep",
            alpha=0.8,
            edgecolor="k",
        )

        title = _(
            "Precision-Recall Trade-off (Mean over {num_seeds} seeds) - {dataset_name}"
        ).format(num_seeds=ctx.cfg.num_seeds, dataset_name=ds_display)
        plt.title(title, fontsize=16)
        plt.xlabel(_("Recall (Sensitivity) - Ability to detect defaults"), fontsize=12)
        plt.ylabel(_("Precision - Trustworthiness of default prediction"), fontsize=12)
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
    language: _LanguageOption = _Language.ENGLISH,
):
    """Analyzes how class imbalance ratio affects model performance.

    Loads each requested dataset independently, injects its imbalance ratio,
    and generates scatter plots to visualize the correlation between imbalance
    ratio and key performance metrics (ROC AUC, F1 Score).
    """
    _setup_i18n(language)
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    metrics = ["roc_auc", "f1_score"]

    # Metric display names mapping
    metric_names = {
        "roc_auc": _("ROC AUC"),
        "f1_score": _("F1 Score"),
    }

    for ds in datasets:
        ds_display = _get_dataset_display(ds.id)
        ctx.logger.info(f"Generating imbalance impact analysis for {ds_display}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        ratio = IMBALANCE_RATIOS.get(ds.id, 1.0)
        df_plot = df.copy()
        df_plot["imbalance_ratio"] = ratio

        available_metrics = [metric for metric in metrics if metric in df_plot.columns]
        if not available_metrics:
            ctx.logger.warning(
                f"No supported metrics found for imbalance analysis in {ds_display}."
            )
            continue

        output_dir = _ensure_output_dir(ctx, ds.id, language)

        plt.figure(figsize=(7 * len(available_metrics), 6))

        for i, metric in enumerate(available_metrics, 1):
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
            plt.xlabel(_("Imbalance Ratio (Majority/Minority) - Log Scale"), fontsize=12)

            metric_display = metric_names.get(metric, metric.replace("_", " ").upper())
            plt.ylabel(metric_display, fontsize=12)

            title = _("{metric_name} vs. Imbalance Ratio - {dataset_name}").format(
                metric_name=metric_display, dataset_name=ds_display
            )
            plt.title(title, fontsize=14)
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
    language: _LanguageOption = _Language.ENGLISH,
):
    """Compares cost-sensitive methods against resampling techniques.

    Generates bar plots to visualize and compare the performance of
    cost-sensitive classifiers (e.g., MetaCost) against various resampling methods.
    """
    _setup_i18n(language)
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    for ds in datasets:
        ds_display = _get_dataset_display(ds.id)
        ctx.logger.info(f"Generating cost-sensitive vs resampling analysis for {ds_display}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.id, language)

        plt.figure(figsize=(12, 8))

        sns.barplot(
            data=df,
            x="technique_display",
            y="roc_auc",
            hue="model_display",
            palette="Set2",
            errorbar="sd",
            capsize=0.1,
        )

        title = _("Cost-Sensitive vs Resampling Performance - {dataset_name}").format(
            dataset_name=ds_display
        )
        plt.title(title, fontsize=16)
        plt.ylabel(_("ROC AUC"), fontsize=12)
        plt.xlabel(_("Technique"), fontsize=12)
        plt.ylim(0.0, 1.0)
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title=_("Model"))
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
    language: _LanguageOption = _Language.ENGLISH,
):
    """Analyzes the effects of hyperparameter choices on model performance.

    Generates heatmaps or line plots to visualize how different hyperparameter
    settings impact key performance metrics like ROC AUC.
    """
    _setup_i18n(language)
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    sns.set_theme(style="whitegrid")

    for ds in datasets:
        ds_display = _get_dataset_display(ds.id)
        ctx.logger.info(f"Generating hyperparameter effects analysis for {ds_display}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        output_dir = _ensure_output_dir(ctx, ds.id, language)

        # Safely parse the 'best_params' string into columns
        # best_params is typically a string representation of a dict: "{'clf__C': 1.0, ...}"
        try:
            hp_df = df["best_params"].apply(lambda x: pd.Series(ast.literal_eval(x)))
        except (ValueError, SyntaxError) as e:
            ctx.logger.error(f"Failed to parse hyperparameters for {ds_display}: {e}")
            continue

        # Combine metrics with extracted hyperparameters
        metric_columns = [
            col
            for col in ("roc_auc", "g_mean", "f1_score", "precision", "recall")
            if col in df.columns
        ]

        # We also need identifiers and display labels for grouping
        display_cols = [col for col in ("technique_display", "model_display") if col in df.columns]
        meta_columns = ["technique", "model", *display_cols]

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
                    hue="technique_display"
                    if "technique_display" in merged_df.columns
                    else "technique",
                    style="model_display" if "model_display" in merged_df.columns else "model",
                    marker="o",
                    errorbar=None,  # Avoid matplotlib warnings on single-sample groups
                )

                if is_numeric:
                    plt.xscale("log")

                title = _("Effect of {param} on ROC AUC - {dataset_name}").format(
                    param=param, dataset_name=ds_display
                )
                plt.title(title, fontsize=16)

                xlabel = _("{param} (Log Scale if numeric)").format(param=param)
                plt.xlabel(xlabel, fontsize=12)
                plt.ylabel(_("ROC AUC"), fontsize=12)
                plt.ylim(0.0, 1.05)
                plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title=_("Technique"))

                filename = output_dir / f"hyperparameter_effects_{param.replace('__', '_')}.png"
                plt.tight_layout()
                plt.savefig(filename, dpi=300)
                plt.close()

                ctx.logger.success(f"Saved hyperparameter effects plot to {filename}")


@app.command("experiment")
def analyze_results(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help=(
                "Identifier of the dataset to analyze. "
                "When omitted, all datasets are analyzed sequentially."
            ),
        ),
    ] = None,
    language: _LanguageOption = _Language.ENGLISH,
):
    """Analyzes overall experimental results in a tabular format."""
    _setup_i18n(language)
    ctx = Context()
    datasets = [dataset] if dataset is not None else list(Dataset)

    # Metrics to report
    metrics = ["roc_auc", "g_mean", "f1_score", "precision", "recall"]

    # Metric display names
    metric_names = {
        "roc_auc": _("ROC AUC"),
        "g_mean": _("G-Mean"),
        "f1_score": _("F1 Score"),
        "precision": _("Precision"),
        "recall": _("Recall"),
    }

    for ds in datasets:
        ds_display = _get_dataset_display(ds.id)
        ctx.logger.info(f"Analyzing results for {ds_display}...")
        df = _load_data(ctx, ds)

        if df.empty:
            continue

        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            ctx.logger.warning(f"No metrics found for {ds_display}.")
            continue

        # Group by Model and Technique
        # We want to aggregate over seeds
        # Ensure display columns exist (they are added in _load_data)
        group_cols = ["model", "technique", "model_display", "technique_display"]

        # Calculate mean and std
        agg_dict = {m: ["mean", "std"] for m in available_metrics}
        summary = df.groupby(group_cols)[available_metrics].agg(agg_dict)

        # Flatten MultiIndex columns
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.reset_index()

        # Create a display DataFrame
        display_df = pd.DataFrame()
        display_df[_("Model")] = summary["model_display"]
        display_df[_("Technique")] = summary["technique_display"]

        for m in available_metrics:
            m_display = metric_names.get(m, m)
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            display_df[m_display] = summary.apply(
                lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}", axis=1
            )

        # Sort by ROC AUC (mean) descending
        sort_col = "roc_auc_mean"
        if sort_col in summary.columns:
            summary = summary.sort_values(sort_col, ascending=False)
            display_df = display_df.iloc[summary.index]

        # Print to console
        print(f"\n=== {ds_display} ===")
        # Use to_string for a simple table representation
        print(display_df.to_string(index=False))
        print("\n")

        # Save to file
        output_dir = _ensure_output_dir(ctx, ds.id, language)
        output_csv = output_dir / "results_summary.csv"
        display_df.to_csv(output_csv, index=False)
        ctx.logger.success(f"Saved summary table to {output_csv}")


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
    language: _LanguageOption = _Language.ENGLISH,
):
    """Runs all analysis commands sequentially."""
    analyze_stability_and_variance(dataset, language=language)
    analyze_risk_tradeoff(dataset, language=language)
    analyze_imbalance_impact(dataset, language=language)
    compare_cost_sensitive_and_resampling(dataset, language=language)
    analyze_hyperparameter_effects(dataset, language=language)
    analyze_results(dataset, language=language)


if __name__ == "__main__":
    app()
