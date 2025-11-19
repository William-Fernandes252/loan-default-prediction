"""
Training module for the credit scoring experiments.

This module handles the training and evaluation of various machine learning models
across different datasets and techniques. It employs parallel processing and
memory mapping to handle large datasets efficiently.
"""

import gc
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
from joblib import Parallel, delayed
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import typer
from typing_extensions import Annotated

from experiments.config import (
    COST_GRIDS,
    CV_FOLDS,
    NUM_SEEDS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    Dataset,
    ModelType,
    Technique,
)
from experiments.utils.jobs import get_safe_jobs

# --- Environment Configuration ---
# Limit thread usage per worker to prevent thread explosion in parallel execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Module Setup ---
MODULE_NAME = "experiments.modeling.train"
if __name__ == "__main__":
    # Fix for joblib pickling when running as __main__
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


# --- Custom Classifiers & Scorers ---


class MetaCostClassifier(ClassifierMixin, BaseEstimator):
    """
    A meta-classifier that makes a base classifier cost-sensitive.

    It uses bagging to estimate class probabilities and then relabels the training
    data to minimize expected cost.
    """

    _estimator_type = "classifier"
    final_estimator_: Optional[BaseEstimator]

    def __init__(
        self,
        base_estimator: BaseEstimator,
        cost_matrix: Optional[Dict[int, Any]] = None,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.cost_matrix = cost_matrix
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.final_estimator_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("MetaCostClassifier only supports binary classification.")

        if self.cost_matrix is None:
            self.final_estimator_ = clone(self.base_estimator).fit(X, y)
            return self

        if not hasattr(self.base_estimator, "predict_proba") and not hasattr(
            self.base_estimator, "decision_function"
        ):
            raise TypeError("Base estimator must support predict_proba or decision_function.")

        # Use Bagging to estimate probabilities
        bagging = BaggingClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=1,
        )
        bagging.fit(X, y)

        # Get probability estimates
        if (
            hasattr(bagging, "oob_decision_function_")
            and bagging.oob_decision_function_ is not None
        ):
            probas = bagging.oob_decision_function_
            # Fallback if OOB score fails (e.g., small sample size)
            if np.any(np.sum(probas, axis=1) == 0):
                probas = bagging.predict_proba(X)
        else:
            probas = bagging.predict_proba(X)

        # Relabel based on expected cost
        # Cost matrix structure: {actual_class: cost_of_error}
        # Assuming binary classification: 0 and 1
        C_FP = self.cost_matrix.get(0, 1)  # Cost of False Positive (predicting 1 when 0)
        C_FN = self.cost_matrix.get(1, 1)  # Cost of False Negative (predicting 0 when 1)

        # Risk of predicting 0: P(1|x) * C_FN
        risk_0 = probas[:, 1] * C_FN
        # Risk of predicting 1: P(0|x) * C_FP
        risk_1 = probas[:, 0] * C_FP

        # Choose class with lower risk
        y_new_np = np.where(risk_1 < risk_0, 1, 0)

        self.final_estimator_ = clone(self.base_estimator).fit(X, y_new_np)
        self.classes_ = self.final_estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.final_estimator_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.final_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.final_estimator_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.final_estimator_.predict_proba(X)


def g_mean_score(y_true, y_pred):
    """Calculates the Geometric Mean of Sensitivity and Specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


g_mean_scorer = make_scorer(g_mean_score)


# --- Model & Pipeline Factories ---


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
    model_type: ModelType, technique: Technique, base_params: dict
) -> List[Dict[str, Any]]:
    """Adjusts the parameter grid based on the technique."""
    new_params = base_params.copy()

    # Handle Cost-Sensitive SVM and Baseline (for comparison)
    if technique == Technique.CS_SVM or (
        technique == Technique.BASELINE
        and model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]
        and technique != Technique.META_COST
    ):
        if technique == Technique.CS_SVM and model_type == ModelType.SVM:
            new_params["clf__class_weight"] = COST_GRIDS

    # Handle MetaCost
    if technique == Technique.META_COST:
        return [{"clf__cost_matrix": COST_GRIDS}]

    return [new_params]


# --- Experiment Execution Logic ---


def _get_checkpoint_path(
    dataset: Dataset, model: ModelType, technique: Technique, seed: int
) -> Path:
    """Generates the path for saving experiment results."""
    temp_dir = RESULTS_DIR / "checkpoints" / dataset.value
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"{model.value}_{technique.value}_seed{seed}.parquet"


def _run_experiment_task(
    dataset_val: str,
    X_mmap_path: str,
    y_mmap_path: str,
    model_type: ModelType,
    technique: Technique,
    seed: int,
) -> Optional[str]:
    """
    Executes a single experiment task (Dataset + Model + Technique + Seed).

    This function is designed to be memory-efficient by using memory-mapped files
    and aggressively clearing memory after training.
    """
    # Reconstruct Enum from string (for pickling safety)
    dataset = Dataset(dataset_val)

    logger.info(
        f"Starting task: {dataset.value} | {model_type.value} | {technique.value} | seed={seed}"
    )

    # 1. Checkpoint Check
    ckpt_path = _get_checkpoint_path(dataset, model_type, technique, seed)
    if ckpt_path.exists():
        return None

    try:
        # 2. Load Data (Memory Mapped - Zero RAM cost initially)
        X_mmap = joblib.load(X_mmap_path, mmap_mode="r")
        y_mmap = joblib.load(y_mmap_path, mmap_mode="r")

        # 3. Validation Checks
        unique, counts = np.unique(y_mmap, return_counts=True)
        min_class_count = counts.min()

        stratify_y = y_mmap
        internal_cv_folds = CV_FOLDS

        if min_class_count < 2:
            stratify_y = None
            internal_cv_folds = max(2, min(CV_FOLDS, min_class_count))

        # 4. Split INDICES only (Negligible RAM usage)
        indices = np.arange(X_mmap.shape[0])
        train_idx, test_idx = train_test_split(
            indices, test_size=0.30, stratify=stratify_y, random_state=seed
        )

        # Validation on the subsets
        y_train_preview = y_mmap[train_idx]
        if len(np.unique(y_train_preview)) < 2:
            return None
        _, train_counts = np.unique(y_train_preview, return_counts=True)
        if train_counts.min() < internal_cv_folds:
            return None
        del y_train_preview

        # 5. MATERIALIZE TRAIN SET (Heavy RAM usage starts here)
        # We only load X_train into RAM. We leave X_test on disk for now.
        X_train = X_mmap[train_idx]
        y_train = y_mmap[train_idx]

        pipeline = build_pipeline(model_type, technique, seed)
        base_grid = model_type.get_params()
        param_grid = get_params_for_technique(model_type, technique, base_grid)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=internal_cv_folds, shuffle=True, random_state=seed),
            n_jobs=1,
            verbose=0,
        )

        # Fit the model
        grid.fit(X_train, y_train)

        # 6. CRITICAL MEMORY SAVER
        # The model is trained. We DO NOT need X_train/y_train anymore.
        # Delete them to free space for X_test.
        del X_train, y_train
        gc.collect()

        # 7. MATERIALIZE TEST SET (Now we have room for this)
        X_test = X_mmap[test_idx]
        y_test = y_mmap[test_idx]

        # 8. Evaluate
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        try:
            y_proba = best_model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
        except (AttributeError, IndexError):
            auc_score = 0.5

        metrics = {
            "dataset": dataset.value,
            "seed": seed,
            "model": model_type.value,
            "technique": technique.value,
            "best_params": str(grid.best_params_),
            "accuracy_balanced": balanced_accuracy_score(y_test, y_pred),
            "g_mean": g_mean_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": auc_score,
        }

        logger.success(
            f"Finished: {dataset.value} | {model_type.value} | {technique.value} | seed={seed} -> "
            f"AUC={auc_score:.4f}, F1={metrics['f1_score']:.4f}"
        )

        # 9. Save Checkpoint
        df_res = pd.DataFrame([metrics])
        df_res.to_parquet(ckpt_path)

        # 10. Final Cleanup
        del grid, best_model, X_test, y_test, X_mmap, y_mmap
        gc.collect()

        return f"{dataset.value} - {model_type.value} - {technique.value} - Seed {seed}"

    except Exception:
        logger.error(
            f"Failed task {dataset.value} {model_type.value} {technique.value} {seed}:\n{traceback.format_exc()}"
        )
        gc.collect()
        return None


def run_dataset_experiments(dataset: Dataset, jobs: int):
    """
    Prepares data and launches parallel experiment tasks for a dataset.
    """
    x_path, y_path = _get_training_artifact_paths(dataset)
    if not x_path.exists() or not y_path.exists():
        logger.error(f"Data missing for {dataset}")
        return

    logger.info(f"Loading data for {dataset.value}...")

    # Load original data
    X_df = pl.read_parquet(x_path).to_pandas()
    y_df = pl.read_parquet(y_path).to_pandas().iloc[:, 0]

    # Create a temporary directory for memory mapping
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create memmap files
        X_mmap_path = os.path.join(temp_dir, "X.mmap")
        y_mmap_path = os.path.join(temp_dir, "y.mmap")

        # Dump data using joblib (optimized for numpy)
        joblib.dump(X_df.to_numpy(), X_mmap_path)
        joblib.dump(y_df.to_numpy(), y_mmap_path)

        # Load back briefly to get shapes/dtypes (sanity check)
        # and to ensure the file is ready
        _ = joblib.load(X_mmap_path, mmap_mode="r")

        # Free the original RAM immediately
        del X_df, y_df
        gc.collect()

        # Generate Task List
        tasks = []
        for seed in range(NUM_SEEDS):
            for model_type in ModelType:
                for technique in Technique:
                    # Skip invalid combinations
                    if technique == Technique.CS_SVM and model_type != ModelType.SVM:
                        continue

                    tasks.append(
                        (
                            dataset.value,
                            X_mmap_path,
                            y_mmap_path,
                            model_type,
                            technique,
                            seed,
                        )
                    )

        logger.info(
            f"Dataset {dataset.value}: Launching {len(tasks)} tasks with {jobs} workers..."
        )

        # Execute Parallel
        Parallel(
            n_jobs=jobs,
            verbose=5,
            pre_dispatch="2*n_jobs",
        )(delayed(_run_experiment_task)(*t) for t in tasks)

    # Consolidate results
    _consolidate_results(dataset)


def _consolidate_results(dataset: Dataset):
    """Combines all checkpoint files into a single results file."""
    ckpt_dir = RESULTS_DIR / "checkpoints" / dataset.value
    all_files = list(ckpt_dir.glob("*.parquet"))

    if all_files:
        logger.info(f"Consolidating {len(all_files)} results for {dataset.value}...")
        df_final = pd.read_parquet(ckpt_dir)
        final_output = RESULTS_DIR / f"{dataset.value}_results.parquet"
        df_final.to_parquet(final_output)
        logger.success(f"Saved consolidated results to {final_output}")
    else:
        logger.warning(f"No results found for {dataset.value}")


def _get_training_artifact_paths(dataset: Dataset) -> Tuple[Path, Path]:
    x_path = PROCESSED_DATA_DIR / f"{dataset.value}_X.parquet"
    y_path = PROCESSED_DATA_DIR / f"{dataset.value}_y.parquet"
    return x_path, y_path


def _artifacts_exist(dataset: Dataset) -> bool:
    x_path, y_path = _get_training_artifact_paths(dataset)
    return x_path.exists() and y_path.exists()


def _get_safe_jobs_for_dataset(dataset: Dataset) -> int:
    """Determines safe number of parallel jobs for the dataset based on its size."""
    size_gb = dataset.get_size_gb()
    safe_jobs = get_safe_jobs(size_gb)
    return safe_jobs


# --- CLI Entry Point ---


@app.command()
def main(
    dataset: Annotated[
        Optional[Dataset],
        typer.Argument(
            help="Dataset to process. If not specified, all datasets will be processed."
        ),
    ] = None,
    jobs: Annotated[
        Optional[int],
        typer.Option(
            "--jobs",
            "-j",
            min=1,
            help="Number of parallel jobs to run. If not specified, a safe number based on dataset size will be used.",
        ),
    ] = None,
):
    """
    Runs the training experiments.
    """
    datasets = [dataset] if dataset is not None else list(Dataset)

    for ds in datasets:
        if not _artifacts_exist(ds):
            logger.warning(f"Artifacts not found for {ds}. Skipping.")
            continue

        gc.collect()
        n_jobs = jobs if jobs is not None else _get_safe_jobs_for_dataset(ds)
        run_dataset_experiments(ds, n_jobs)


if __name__ == "__main__":
    app()
