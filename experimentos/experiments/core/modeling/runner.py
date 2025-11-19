"""Experiment running logic for executing modeling tasks with memory efficiency."""

import gc
from pathlib import Path
import traceback
from typing import Any, Optional

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from experiments.core.data import Dataset
from experiments.core.modeling import (
    ModelType,
    Technique,
    build_pipeline,
    g_mean_score,
    get_hyperparameters,
    get_params_for_technique,
)


def run_experiment_task(
    dataset_val: str,
    X_mmap_path: str,
    y_mmap_path: str,
    model_type: ModelType,
    technique: Technique,
    cost_grids: dict[str, Any],
    cv_folds: int,
    checkpoint_path: Path,
    seed: int,
) -> Optional[str]:
    """
    Executes a single experiment task (Dataset + Model + Technique + Seed).

    This function is designed to be memory-efficient by using memory-mapped files
    and aggressively clearing memory after training.

    Args:
        dataset_val (str): The dataset identifier as a string.
        X_mmap_path (str): Path to the memory-mapped feature data.
        y_mmap_path (str): Path to the memory-mapped target data.
        model_type (ModelType): The type of model to use.
        technique (Technique): The technique to apply (e.g., cost-sensitive).
        cost_grids (dict[str, Any]): Cost grids for cost-sensitive techniques.
        cv_folds (int): Number of cross-validation folds.
        checkpoint_path (Path): Path to save the experiment checkpoint.
        seed (int): Random seed for reproducibility.
    """
    # Reconstruct Enum from string (for pickling safety)
    dataset = Dataset(dataset_val)

    logger.info(
        f"Starting task: {dataset.value} | {model_type.value} | {technique.value} | seed={seed}"
    )

    # 1. Checkpoint Check
    if checkpoint_path.exists():
        return None

    try:
        # 2. Load Data (Memory Mapped - Zero RAM cost initially)
        X_mmap = joblib.load(X_mmap_path, mmap_mode="r")
        y_mmap = joblib.load(y_mmap_path, mmap_mode="r")

        # 3. Validation Checks
        unique, counts = np.unique(y_mmap, return_counts=True)
        min_class_count = counts.min()

        stratify_y = y_mmap
        internal_cv_folds = cv_folds

        if min_class_count < 2:
            stratify_y = None
            internal_cv_folds = max(2, min(cv_folds, min_class_count))

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
        base_grid = get_hyperparameters(model_type)
        param_grid = get_params_for_technique(model_type, technique, base_grid, cost_grids)

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
        df_res.to_parquet(checkpoint_path)

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
