"""Factory functions and pipeline builders for machine learning models and techniques."""

import enum
from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.core.modeling.estimators import MetaCostClassifier


class ModelType(enum.Enum):
    """Types of models used in experiments."""

    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    ADA_BOOST = "ada_boost"
    MLP = "mlp"


class Technique(enum.Enum):
    """Types of techniques used in experiments."""

    BASELINE = "baseline"
    SMOTE = "smote"
    RANDOM_UNDER_SAMPLING = "random_under_sampling"
    SMOTE_TOMEK = "smote_tomek"
    META_COST = "meta_cost"
    CS_SVM = "cs_svm"


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
        ModelType.ADA_BOOST: {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 1.0],
        },
        ModelType.MLP: {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [0.0001, 0.01],
            "clf__max_iter": [500],  # Ensure convergence
            "clf__early_stopping": [True],
        },
    }
    return params.get(model_type, {})


def get_model_instance(model_type: ModelType, random_state: int) -> BaseEstimator:
    """Factory function to create model instances."""
    if model_type == ModelType.SVM:
        return SVC(random_state=random_state, probability=True)
    elif model_type == ModelType.RANDOM_FOREST:
        return RandomForestClassifier(random_state=random_state, n_jobs=1)
    elif model_type == ModelType.ADA_BOOST:
        return AdaBoostClassifier(random_state=random_state, algorithm="SAMME")
    elif model_type == ModelType.MLP:
        return MLPClassifier(random_state=random_state)
    else:
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
    cost_matrix: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Adjusts the parameter grid based on the technique."""
    new_params = base_params.copy()

    # Handle Cost-Sensitive SVM and Baseline (for comparison)
    if technique == Technique.CS_SVM or (
        technique == Technique.BASELINE
        and model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]
        and technique != Technique.META_COST
    ):
        if technique == Technique.CS_SVM and model_type == ModelType.SVM:
            new_params["clf__class_weight"] = cost_matrix

    # Handle MetaCost
    if technique == Technique.META_COST:
        return [{"clf__cost_matrix": cost_matrix}]
    return [new_params]
