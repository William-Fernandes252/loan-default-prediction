"""Factory functions and pipeline builders for machine learning models and techniques."""

import enum
from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiments.core.choices import Choice
from experiments.core.modeling.estimators import MetaCostClassifier
from experiments.core.modeling.estimators import RobustSVC as SVC
from experiments.core.modeling.estimators import RobustXGBClassifier as XGBClassifier


class ModelType(enum.Enum):
    """Types of models used in experiments."""

    RANDOM_FOREST = Choice("random_forest", "Random Forest")
    SVM = Choice("svm", "Support Vector Machine")
    XGBOOST = Choice("xgboost", "XGBoost")
    MLP = Choice("mlp", "Multi-Layer Perceptron")

    def __str__(self) -> str:
        return self.id

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def display_name(self) -> str:
        return self.value.display_name

    @classmethod
    def from_id(cls, identifier: str) -> "ModelType":
        for member in cls:
            if member.id == identifier:
                return member
        raise ValueError(f"Unknown model type id: {identifier}")

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        if isinstance(value, Choice):
            for member in cls:
                if member.value == value:
                    return member
        if isinstance(value, str):
            for member in cls:
                if member.id == value or member.name.lower() == value.lower():
                    return member
        return None


class Technique(enum.Enum):
    """Types of techniques used in experiments."""

    BASELINE = Choice("baseline", "Baseline")
    SMOTE = Choice("smote", "SMOTE")
    RANDOM_UNDER_SAMPLING = Choice("random_under_sampling", "Random Under Sampling")
    SMOTE_TOMEK = Choice("smote_tomek", "SMOTE Tomek")
    META_COST = Choice("meta_cost", "Meta Cost")
    CS_SVM = Choice("cs_svm", "Cost-sensitive SVM")

    def __str__(self) -> str:
        return self.id

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def display_name(self) -> str:
        return self.value.display_name

    @classmethod
    def from_id(cls, identifier: str) -> "Technique":
        for member in cls:
            if member.id == identifier:
                return member
        raise ValueError(f"Unknown technique id: {identifier}")

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        if isinstance(value, Choice):
            for member in cls:
                if member.value == value:
                    return member
        if isinstance(value, str):
            for member in cls:
                if member.id == value or member.name.lower() == value.lower():
                    return member
        return None


def get_hyperparameters(model_type: ModelType) -> dict:
    """Get hyperparameters for the given model type."""
    params: dict[ModelType, dict[str, Any]] = {
        ModelType.SVM: {
            "clf__C": [0.1, 1, 10, 100],
            "clf__kernel": ["rbf"],  # Linear can be too slow for large datasets
            "clf__probability": [True],  # Necessary for some metrics or MetaCost
            # For CS-SVM, the weight will be injected dynamically or via grid here
            # If it is Baseline, class_weight is None.
        },
        ModelType.RANDOM_FOREST: {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 5],
            # Random Forest also accepts class_weight, useful for comparison with CS-SVM
        },
        ModelType.XGBOOST: {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.3],  # 'eta' in XGBoost terminology
            "clf__max_depth": [3, 6, 10],  # Important for controlling complexity
            "clf__subsample": [0.8, 1.0],  # Helps to avoid over-fitting
            "clf__colsample_bytree": [0.8, 1.0],  # Fraction of features per tree
        },
        ModelType.MLP: {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [0.0001, 0.01],
            "clf__early_stopping": [True],
        },
    }
    return params.get(model_type, {})


def get_model_instance(model_type: ModelType, random_state: int) -> BaseEstimator:
    """Factory function to create model instances with ROBUST DEFAULTS."""
    if model_type == ModelType.SVM:
        return SVC(random_state=random_state, probability=True)
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(random_state=random_state, n_jobs=1)
    elif model_type == ModelType.XGBOOST:
        return XGBClassifier(
            random_state=random_state, eval_metric="logloss", n_jobs=1, objective="binary:logistic"
        )
    elif model_type == ModelType.MLP:
        return MLPClassifier(
            random_state=random_state, max_iter=1000, early_stopping=True, n_iter_no_change=20
        )
    raise ValueError(f"Unknown model: {model_type}")


def build_pipeline(model_type: ModelType, technique: Technique, random_state: int) -> ImbPipeline:
    """Constructs the training pipeline including preprocessing and sampling."""
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

    # Add sampling technique if applicable
    if technique == Technique.RANDOM_UNDER_SAMPLING:
        steps.append(("sampler", RandomUnderSampler(random_state=random_state)))
    elif technique == Technique.SMOTE:
        steps.append(("sampler", SMOTE(random_state=random_state)))
    elif technique == Technique.SMOTE_TOMEK:
        steps.append(("sampler", SMOTETomek(random_state=random_state)))

    # Add classifier
    clf = get_model_instance(model_type, random_state)

    # Wrap with MetaCost if applicable
    if technique == Technique.META_COST:
        clf = MetaCostClassifier(base_estimator=clf, random_state=random_state)

    steps.append(("clf", clf))
    return ImbPipeline(steps)


def get_params_for_technique(
    model_type: ModelType,
    technique: Technique,
    base_params: dict,
    cost_matrix: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Adjusts the parameter grid based on the technique."""
    new_params: dict[str, Any] = base_params.copy()

    # 1. Handle CS-SVM (Cost-Sensitive SVM)
    if technique == Technique.CS_SVM or (
        technique == Technique.BASELINE
        and model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]
        and technique != Technique.META_COST
    ):
        if technique == Technique.CS_SVM and model_type == ModelType.SVM:
            new_params["clf__class_weight"] = cost_matrix

    # 2. Handle MetaCost
    if technique == Technique.META_COST:
        # Start with the cost matrix (parameter of MetaCost itself)
        meta_cost_params = {"clf__cost_matrix": cost_matrix}

        # Iterate through the base parameters (e.g., learning_rate, hidden_layer_sizes)
        # and "push" them down into the base_estimator
        for key, value in base_params.items():
            if key.startswith("clf__"):
                # Transform: clf__param -> clf__base_estimator__param
                new_key = key.replace("clf__", "clf__base_estimator__", 1)
                meta_cost_params[new_key] = value
            else:
                # Keep non-classifier params (e.g. sampler params) as is
                meta_cost_params[key] = value

        return [meta_cost_params]

    return [new_params]
