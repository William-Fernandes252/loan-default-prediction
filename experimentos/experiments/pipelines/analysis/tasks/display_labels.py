from typing import Any

from experiments.core.analysis.metrics import Metric
from experiments.core.data import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.pipelines.analysis.pipeline import AnalysisPipelineContext

_DATASET_DISPLAY_NAMES: dict[Dataset, str] = {
    Dataset.CORPORATE_CREDIT_RATING: "Corporate Credit Rating",
    Dataset.LENDING_CLUB: "Lending Club",
    Dataset.TAIWAN_CREDIT: "Taiwan Credit",
}

_METRIC_DISPLAY_NAMES: dict[Metric, str] = {
    Metric.BALANCED_ACCURACY: "Balanced Accuracy",
    Metric.G_MEAN: "G-Mean",
    Metric.F1_SCORE: "F1 Score",
    Metric.PRECISION: "Precision",
    Metric.SENSITIVITY: "Sensitivity",
}

_MODEL_TYPE_DISPLAY_NAMES: dict[str, str] = {
    ModelType.RANDOM_FOREST.value: "Random Forest",
    ModelType.SVM.value: "SVM",
    ModelType.XGBOOST.value: "XGBoost",
    ModelType.MLP.value: "MLP",
}

_TECHNIQUE_DISPLAY_NAMES: dict[str, str] = {
    Technique.BASELINE.value: "Baseline",
    Technique.SMOTE.value: "SMOTE",
    Technique.RANDOM_UNDER_SAMPLING.value: "RUS",
    Technique.SMOTE_TOMEK.value: "SMOTE-Tomek",
    Technique.CS_SVM.value: "CS-SVM",
    "meta_cost": "Meta Cost",
}


def translate(context: AnalysisPipelineContext, msgid: str, **kwargs: str) -> str:
    if context.translator is not None:
        return context.translator.translate(msgid, **kwargs)
    if kwargs:
        return msgid.format(**kwargs)
    return msgid


def _resolve_display_name(value: Any, display_names: dict[str, str]) -> str:
    raw_value = value.value if hasattr(value, "value") else str(value)
    return display_names.get(raw_value, raw_value.replace("_", " ").title())


def get_dataset_display_name(context: AnalysisPipelineContext, dataset: Dataset) -> str:
    display_name = _DATASET_DISPLAY_NAMES.get(dataset, dataset.value)
    return translate(context, display_name)


def get_metric_display_name(context: AnalysisPipelineContext, metric: Metric) -> str:
    display_name = _METRIC_DISPLAY_NAMES.get(metric, metric.value.replace("_", " ").title())
    return translate(context, display_name)


def get_model_type_display_name(
    context: AnalysisPipelineContext, model_type: ModelType | str
) -> str:
    return translate(context, _resolve_display_name(model_type, _MODEL_TYPE_DISPLAY_NAMES))


def get_technique_display_name(
    context: AnalysisPipelineContext, technique: Technique | str
) -> str:
    return translate(context, _resolve_display_name(technique, _TECHNIQUE_DISPLAY_NAMES))


def add_display_label_columns(pdf: Any, context: AnalysisPipelineContext) -> Any:
    display_pdf = pdf.copy()

    if "technique" in display_pdf.columns:
        display_pdf["technique_label"] = display_pdf["technique"].map(
            lambda value: get_technique_display_name(context, value)
        )

    if "model_type" in display_pdf.columns:
        display_pdf["model_type_label"] = display_pdf["model_type"].map(
            lambda value: get_model_type_display_name(context, value)
        )

    return display_pdf


def create_plot_display_dataframe(
    pdf: Any, context: AnalysisPipelineContext
) -> tuple[Any, str, str]:
    display_pdf = add_display_label_columns(pdf, context)
    technique_column = translate(context, "Technique")
    model_column = translate(context, "Model")

    rename_map: dict[str, str] = {}
    if "technique_label" in display_pdf.columns:
        rename_map["technique_label"] = technique_column
    if "model_type_label" in display_pdf.columns:
        rename_map["model_type_label"] = model_column

    if rename_map:
        display_pdf = display_pdf.rename(columns=rename_map)

    return display_pdf, technique_column, model_column


def create_export_dataframe(pdf: Any, context: AnalysisPipelineContext) -> Any:
    export_pdf = pdf.copy()

    if "technique" in export_pdf.columns:
        export_pdf["technique"] = export_pdf["technique"].map(
            lambda value: get_technique_display_name(context, value)
        )

    if "model_type" in export_pdf.columns:
        export_pdf["model_type"] = export_pdf["model_type"].map(
            lambda value: get_model_type_display_name(context, value)
        )

    rename_map: dict[str, str] = {}
    if "technique" in export_pdf.columns:
        rename_map["technique"] = translate(context, "Technique")
    if "model_type" in export_pdf.columns:
        rename_map["model_type"] = translate(context, "Model")
    if rename_map:
        export_pdf = export_pdf.rename(columns=rename_map)

    return export_pdf


__all__ = [
    "add_display_label_columns",
    "create_export_dataframe",
    "create_plot_display_dataframe",
    "get_dataset_display_name",
    "get_metric_display_name",
    "get_model_type_display_name",
    "get_technique_display_name",
    "translate",
]
