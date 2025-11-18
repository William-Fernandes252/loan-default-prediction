import gc
from pathlib import Path
import sys
from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, cpu_count, delayed
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

MODULE_NAME = "experiments.modeling.train"
if __name__ == "__main__":
    sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])

app = typer.Typer()


# region --- 1. MetaCost Classifier (unchanged)
class MetaCostClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    final_estimator_: BaseEstimator | None

    def __init__(
        self,
        base_estimator: BaseEstimator,
        cost_matrix: dict[int, Any] | None = None,
        n_estimators: int = 50,
        random_state: int | None = None,
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

        bagging = BaggingClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=1,
        )
        bagging.fit(X, y)

        if (
            hasattr(bagging, "oob_decision_function_")
            and bagging.oob_decision_function_ is not None
        ):
            probas = bagging.oob_decision_function_
            if np.any(np.sum(probas, axis=1) == 0):
                probas = bagging.predict_proba(X)
        else:
            probas = bagging.predict_proba(X)

        C_FP = self.cost_matrix.get(0, 1)
        C_FN = self.cost_matrix.get(1, 1)
        risk_0 = probas[:, 1] * C_FN
        risk_1 = probas[:, 0] * C_FP
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


g_mean_scorer = make_scorer(g_mean_score)


def get_model_instance(model_type: ModelType, random_state: int) -> BaseEstimator:
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
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if technique == Technique.RANDOM_UNDER_SAMPLING:
        steps.append(("sampler", RandomUnderSampler(random_state=random_state)))
    elif technique == Technique.SMOTE:
        steps.append(("sampler", SMOTE(random_state=random_state)))
    elif technique == Technique.SMOTE_TOMEK:
        steps.append(("sampler", SMOTETomek(random_state=random_state)))
    clf = get_model_instance(model_type, random_state)
    if technique == Technique.META_COST:
        clf = MetaCostClassifier(base_estimator=clf, random_state=random_state)
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def get_params_for_technique(
    model_type: ModelType, technique: Technique, base_params: dict
) -> list[dict]:
    new_params = base_params.copy()
    if technique == Technique.CS_SVM or (
        technique == Technique.BASELINE
        and model_type in [ModelType.SVM, ModelType.RANDOM_FOREST]
        and technique != Technique.META_COST
    ):
        if technique == Technique.CS_SVM and model_type == ModelType.SVM:
            new_params["clf__class_weight"] = COST_GRIDS
    if technique == Technique.META_COST:
        return [{"clf__cost_matrix": COST_GRIDS}]
    return [new_params]


# endregion


# region --- 4. ATOMIC Training Logic
def _get_checkpoint_path(
    dataset: Dataset, model: ModelType, technique: Technique, seed: int
) -> Path:
    temp_dir = RESULTS_DIR / "checkpoints" / dataset.value
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"{model.value}_{technique.value}_seed{seed}.parquet"


def run_atomic_task(
    dataset_val: str,
    X: np.ndarray,  # <--- Changed type hint to np.ndarray
    y: np.ndarray,  # <--- Changed type hint to np.ndarray
    model_type: ModelType,
    technique: Technique,
    seed: int,
):
    # Reconstruct Enum
    dataset = Dataset(dataset_val)

    # 1. Checkpoint Check
    ckpt_path = _get_checkpoint_path(dataset, model_type, technique, seed)
    if ckpt_path.exists():
        return None

    try:
        # 2. Convert Y to Series for some Stratification util checks if needed,
        # but generally sklearn works fine with numpy arrays.
        # We use pandas helper just for value_counts logic, but we can do it with numpy too.
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()

        stratify_y: np.ndarray | None = y
        internal_cv_folds = CV_FOLDS

        if min_class_count < 2:
            stratify_y = None
            internal_cv_folds = max(2, min(CV_FOLDS, min_class_count))

        # 3. Split (Using Numpy Arrays)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=stratify_y, random_state=seed
        )

        # Validation Checks
        if len(np.unique(y_train)) < 2:
            return None

        # Check minority count in train
        _, train_counts = np.unique(y_train, return_counts=True)
        if train_counts.min() < internal_cv_folds:
            return None

        # 4. Build & Fit
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
        grid.fit(X_train, y_train)

        # 5. Evaluate
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

        # 6. SAVE CHECKPOINT
        df_res = pd.DataFrame([metrics])
        df_res.to_parquet(ckpt_path)

        # 7. FORCE GARBAGE COLLECTION
        # This helps clear intermediate arrays created during GridSearch
        del grid, pipeline, X_train, X_test, y_train, y_test
        gc.collect()

        return f"{dataset.value} - {model_type.value} - {technique.value} - Seed {seed}"

    except Exception as e:
        logger.error(
            f"Failed task {dataset.value} {model_type.value} {technique.value} {seed}: {e}"
        )
        # Force GC on exception too
        gc.collect()
        return None


def process_dataset_parallel(dataset: Dataset, jobs: int):
    x_path, y_path = _get_training_artifact_paths(dataset)
    if not x_path.exists() or not y_path.exists():
        logger.error(f"Data missing for {dataset}")
        return

    logger.info(f"Loading data for {dataset.value}...")
    # Load as Pandas
    X_df = pl.read_parquet(x_path).to_pandas()
    y_df = pl.read_parquet(y_path).to_pandas().iloc[:, 0]

    # CRITICAL FIX: Convert to Numpy Arrays immediately
    # Joblib can memory-map Numpy arrays (share RAM), but it pickles Pandas (copies RAM).
    X = np.ascontiguousarray(X_df.to_numpy())
    y = np.ascontiguousarray(y_df.to_numpy())

    # Delete the pandas objects to free RAM before forking
    del X_df, y_df
    gc.collect()

    # Generate Task List
    tasks = []
    for seed in range(NUM_SEEDS):
        for model_type in ModelType:
            for technique in Technique:
                if technique == Technique.CS_SVM and model_type != ModelType.SVM:
                    continue
                tasks.append((dataset.value, X, y, model_type, technique, seed))

    logger.info(f"Dataset {dataset.value}: Launching {len(tasks)} tasks with {jobs} workers...")

    # Execute Parallel
    # max_nbytes=None allows joblib to mmap arrays of any size
    # pre_dispatch='2*n_jobs' prevents memory from filling up with queued tasks
    Parallel(
        n_jobs=jobs,
        verbose=5,
        max_nbytes=None,
        pre_dispatch="2*n_jobs",
    )(delayed(run_atomic_task)(*t) for t in tasks)

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


# endregion


# region --- 5. Helpers (unchanged)
def _get_training_artifact_paths(dataset: Dataset) -> tuple[Path, Path]:
    x_path = PROCESSED_DATA_DIR / f"{dataset.value}_X.parquet"
    y_path = PROCESSED_DATA_DIR / f"{dataset.value}_y.parquet"
    return x_path, y_path


def _artifacts_exist(dataset: Dataset) -> bool:
    x_path, y_path = _get_training_artifact_paths(dataset)
    return x_path.exists() and y_path.exists()


# endregion


# region --- 6. CLI Entry Point
@app.command()
def main(
    dataset: Annotated[Dataset | None, typer.Argument()] = None,
    jobs: Annotated[int | None, typer.Option("--jobs", "-j", min=1)] = None,
):
    datasets = [dataset] if dataset is not None else list(Dataset)

    # Determine CPU Count
    # SAFEGUARD: If jobs not specified, default to 50% of CPUs to be safe on RAM
    available_cpus = cpu_count()

    if jobs is None:
        n_jobs = max(1, available_cpus // 2)  # Conservative default
    else:
        n_jobs = jobs

    logger.info(f"Global Parallelism: Using {n_jobs} workers.")

    for ds in datasets:
        if not _artifacts_exist(ds):
            continue

        # Force GC between datasets
        gc.collect()
        process_dataset_parallel(ds, n_jobs)


if __name__ == "__main__":
    app()
