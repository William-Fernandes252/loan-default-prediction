"""Experiment running logic refactored for cohesion and testability."""

import gc
from pathlib import Path
import traceback
from typing import Any, Dict, Optional, Tuple

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from experiments.core.data import Dataset
from experiments.core.modeling.factories import (
    build_pipeline,
    get_hyperparameters,
    get_params_for_technique,
)
from experiments.core.modeling.metrics import g_mean_score
from experiments.core.modeling.schema import ExperimentConfig
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.models import ModelVersioningService


def _load_and_validate_data(
    X_mmap_path: str, y_mmap_path: str, seed: int, cv_folds: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads memory-mapped data, validates class distribution, and performs the split.
    Returns (X_train, y_train, X_test, y_test) or None if validation fails.
    """
    X_mmap = joblib.load(X_mmap_path, mmap_mode="r")
    y_mmap = joblib.load(y_mmap_path, mmap_mode="r")

    # Validation: Check minimum class counts
    _, counts = np.unique(y_mmap, return_counts=True)
    if counts.min() < 2:
        return None

    # Determine stratification
    stratify_y = y_mmap if counts.min() >= cv_folds else None

    indices = np.arange(X_mmap.shape[0])
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, stratify=stratify_y, random_state=seed
    )

    # Validate training split sufficiency
    y_train_preview = y_mmap[train_idx]
    if len(np.unique(y_train_preview)) < 2:
        return None

    _, train_counts = np.unique(y_train_preview, return_counts=True)
    if train_counts.min() < cv_folds:
        return None

    # Materialize training data (Test data remains indices/mmap until needed to save RAM)
    return X_mmap[train_idx], y_mmap[train_idx], X_mmap[test_idx], y_mmap[test_idx]


def _train_and_optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: ModelType,
    technique: Technique,
    seed: int,
    cv_folds: int,
    cost_grids: Any,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Builds the pipeline and runs GridSearchCV."""

    pipeline = build_pipeline(model_type, technique, seed)
    base_grid = get_hyperparameters(model_type)
    param_grid = get_params_for_technique(model_type, technique, base_grid, cost_grids)

    # Adjust CV if class count is low
    _, counts = np.unique(y_train, return_counts=True)
    actual_folds = max(2, min(cv_folds, counts.min()))

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed),
        n_jobs=1,
        verbose=0,
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def _evaluate_model(
    model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float | str]:
    """Calculates all performance metrics."""
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        auc_score = 0.5

    return {
        "accuracy_balanced": balanced_accuracy_score(y_test, y_pred),
        "g_mean": g_mean_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": auc_score,
    }


def run_experiment_task(
    cfg: ExperimentConfig,
    dataset_val: str,
    X_mmap_path: str,
    y_mmap_path: str,
    model_type: ModelType,
    technique: Technique,
    seed: int,
    checkpoint_path: Path,
    model_versioning_service: Optional[ModelVersioningService] = None,
) -> Optional[str]:
    """Orchestrates the experiment steps: Load -> Train -> Evaluate -> Save."""

    # 1. Setup & Checkpoint Check
    try:
        dataset = Dataset(dataset_val)
    except ValueError:
        dataset = Dataset.from_id(str(dataset_val))

    if checkpoint_path.exists():
        if cfg.discard_checkpoints:
            checkpoint_path.unlink(missing_ok=True)
        else:
            logger.info(f"Skipping existing: {dataset.id} | seed={seed}")
            return None

    logger.info(f"Starting: {dataset.display_name} | {model_type.name} | {technique.name}")

    try:
        # 2. Data Preparation
        data_split = _load_and_validate_data(X_mmap_path, y_mmap_path, seed, cfg.cv_folds)
        if data_split is None:
            return None

        X_train, y_train, X_test, y_test = data_split

        # 3. Training
        best_model, best_params = _train_and_optimize(
            X_train, y_train, model_type, technique, seed, cfg.cv_folds, cfg.cost_grids
        )

        # Free training memory immediately
        del X_train, y_train
        gc.collect()

        # 4. Evaluation
        metrics = _evaluate_model(best_model, X_test, y_test)

        # Add metadata
        metrics.update(
            {
                "dataset": dataset.id,
                "seed": seed,
                "model": model_type.id,
                "technique": technique.id,
                "best_params": str(best_params),
            }
        )

        logger.success(
            f"Done: {dataset.display_name} | {model_type.name} | seed={seed} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )

        # 5. Persistence
        if model_versioning_service:
            try:
                model_versioning_service.save_model(best_model, None)
            except Exception as e:
                logger.warning(f"Model save failed: {e}")

        pd.DataFrame([metrics]).to_parquet(checkpoint_path)

        return f"{dataset.id}-{model_type.id}-{seed}"

    except Exception:
        logger.error(f"Failed task: {traceback.format_exc()}")
        return None
    finally:
        # Final cleanup
        gc.collect()
