"""CLI for analyzing experimental results.

This module provides CLI commands for various analysis types using
a pipeline-based architecture with dependency injection.
"""

import enum
import gettext
from pathlib import Path
import sys
from typing import Optional

import typer
from typing_extensions import Annotated

from experiments.context import Context
from experiments.core.analysis.pipeline import AnalysisPipelineFactory
from experiments.core.analysis.protocols import TranslationFunc
from experiments.core.data import Dataset


class _Language(enum.Enum):
    """Supported languages for analysis reports."""

    ENGLISH = "en_US"
    PORTUGUESE_BRAZIL = "pt_BR"


# Type alias for typer language option
_LanguageOption = Annotated[
    _Language,
    typer.Option("--language", "-l", help="Language code for translations"),
]

# Type alias for optional dataset argument
_DatasetArgument = Annotated[
    Optional[Dataset],
    typer.Argument(
        help=(
            "Identifier of the dataset to analyze. "
            "When omitted, all datasets are analyzed sequentially."
        ),
    ),
]


def _noop(s: str) -> str:
    """No-op translation function (returns input unchanged)."""
    return s


def _setup_i18n(language: _Language) -> TranslationFunc:
    """Set up gettext translation based on the language code.

    Args:
        language: The language to use for translations.

    Returns:
        A translation function that translates strings.
    """
    locales_dir = Path(__file__).parents[2] / "locales"
    try:
        lang = gettext.translation(
            "base",
            localedir=locales_dir,
            languages=[language.value],
        )
        lang.install()
        return lang.gettext
    except FileNotFoundError:
        return _noop


class _ContextOutputPathProvider:
    """Provides output paths using Context configuration.

    Implements the OutputPathProvider protocol using the application
    context to determine output directories based on language setting.
    """

    def __init__(self, ctx: Context, language: _Language) -> None:
        """Initialize the provider.

        Args:
            ctx: Application context.
            language: Language for path resolution.
        """
        self._ctx = ctx
        self._language = language

    def get_output_dir(
        self,
        dataset_id: str,
        is_figure: bool = True,
    ) -> Path:
        """Get the output directory for a dataset.

        Args:
            dataset_id: The dataset identifier.
            is_figure: Whether the output is a figure.

        Returns:
            The path to the output directory.
        """
        base_dir = self._ctx.cfg.figures_dir.parent / self._language.value
        if is_figure:
            output_dir = base_dir / dataset_id / "figures"
        else:
            output_dir = base_dir / dataset_id

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


def _create_pipeline_factory(
    ctx: Context,
    language: _Language,
    translate: TranslationFunc,
) -> AnalysisPipelineFactory:
    """Create a configured pipeline factory.

    Args:
        ctx: Application context.
        language: Language for output paths.
        translate: Translation function.

    Returns:
        A configured AnalysisPipelineFactory.
    """
    output_provider = _ContextOutputPathProvider(ctx, language)
    return AnalysisPipelineFactory(
        path_provider=ctx,
        output_path_provider=output_provider,
        translate=translate,
    )


def _resolve_datasets(dataset: Dataset | None) -> list[Dataset]:
    """Resolve a single dataset to a list, defaulting to all datasets.

    Args:
        dataset: Optional single dataset.

    Returns:
        List of datasets to process.
    """
    if dataset is not None:
        return [dataset]
    return list(Dataset)


MODULE_NAME = "experiments.cli.analysis"

if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])


app = typer.Typer()


@app.command("stability")
def analyze_stability_and_variance(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Analyze the stability and variance of model performance across seeds.

    Generates Boxplots grouped by Technique and Model to visualize:
    1. Performance spread (variance): How much results change with random seeds?
    2. Median performance: Which method is consistently better?
    3. Outliers: Cases where the model failed significantly.
    """
    translate = _setup_i18n(language)
    ctx = Context()
    factory = _create_pipeline_factory(ctx, language, translate)

    pipeline = factory.create_stability_pipeline()
    datasets = _resolve_datasets(dataset)

    for ds in datasets:
        ctx.logger.info(f"Generating stability analysis for {translate(ds.display_name)}...")
        pipeline.run(ds)


@app.command("risktradeoff")
def analyze_risk_tradeoff(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Analyze the risk trade-off curves for different models and techniques.

    In credit risk, a false negative (not identifying a defaulter) is costly.
    However, rejecting many good payers (false positives) results in lost revenue.
    Which technique offered the best Recall without destroying Precision?
    The F1-Score helps summarize this, but the graph tells the full story.
    """
    translate = _setup_i18n(language)
    ctx = Context()
    factory = _create_pipeline_factory(ctx, language, translate)

    pipeline = factory.create_risk_tradeoff_pipeline()
    datasets = _resolve_datasets(dataset)

    for ds in datasets:
        ctx.logger.info(f"Generating risk trade-off analysis for {translate(ds.display_name)}...")
        pipeline.run(ds)


@app.command("imbalanceimpact")
def analyze_imbalance_impact(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Analyze how class imbalance ratio affects model performance.

    Loads each requested dataset independently, injects its imbalance ratio,
    and generates scatter plots to visualize the correlation between imbalance
    ratio and key performance metrics (Balanced Accuracy, F1-Score, G-mean and Sensitivity).
    """
    translate = _setup_i18n(language)
    ctx = Context()
    factory = _create_pipeline_factory(ctx, language, translate)

    pipeline = factory.create_imbalance_impact_pipeline()
    datasets = _resolve_datasets(dataset)

    for ds in datasets:
        ctx.logger.info(
            f"Generating imbalance impact analysis for {translate(ds.display_name)}..."
        )
        pipeline.run(ds)


@app.command("csvsresampling")
def compare_cost_sensitive_and_resampling(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Compare cost-sensitive methods against resampling techniques.

    Generates bar plots to visualize and compare the performance of
    cost-sensitive classifiers (e.g., MetaCost) against various resampling methods.
    """
    translate = _setup_i18n(language)
    ctx = Context()
    factory = _create_pipeline_factory(ctx, language, translate)

    pipeline = factory.create_cost_sensitive_vs_resampling_pipeline()
    datasets = _resolve_datasets(dataset)

    for ds in datasets:
        ctx.logger.info(
            f"Generating cost-sensitive vs resampling analysis for {translate(ds.display_name)}..."
        )
        pipeline.run(ds)


@app.command("hyperparameters")
def analyze_hyperparameter_effects(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Analyze the effects of hyperparameter choices on model performance.

    Generates heatmaps or line plots to visualize how different hyperparameter
    settings impact key performance metrics, like Balanced Accuracy and G-mean
    across models and techniques.
    """
    translate = _setup_i18n(language)
    ctx = Context()
    factory = _create_pipeline_factory(ctx, language, translate)

    pipeline = factory.create_hyperparameter_pipeline()
    datasets = _resolve_datasets(dataset)

    for ds in datasets:
        ctx.logger.info(
            f"Generating hyperparameter effects analysis for {translate(ds.display_name)}..."
        )
        pipeline.run(ds)


@app.command("experiment")
def analyze_results(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Analyze overall experimental results in a tabular format.

    Generates summary tables with mean and standard deviation for each metric,
    exported as CSV, LaTeX, and PNG table images.
    """
    translate = _setup_i18n(language)
    ctx = Context()
    factory = _create_pipeline_factory(ctx, language, translate)

    pipeline = factory.create_experiment_summary_pipeline()
    datasets = _resolve_datasets(dataset)

    for ds in datasets:
        ctx.logger.info(f"Analyzing results for {translate(ds.display_name)}...")
        pipeline.run(ds)


@app.command("all")
def run_all_analyses(
    dataset: _DatasetArgument = None,
    language: _LanguageOption = _Language.ENGLISH,
) -> None:
    """Run all analysis commands sequentially."""
    analyze_stability_and_variance(dataset, language=language)
    analyze_risk_tradeoff(dataset, language=language)
    analyze_imbalance_impact(dataset, language=language)
    compare_cost_sensitive_and_resampling(dataset, language=language)
    analyze_hyperparameter_effects(dataset, language=language)
    analyze_results(dataset, language=language)


if __name__ == "__main__":
    app()
