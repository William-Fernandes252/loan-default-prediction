"""Experiment running logic for executing modeling tasks with memory efficiency."""

import gc
from pathlib import Path
import traceback
from typing import Optional

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
from experiments.core.modeling.factories import (
    build_pipeline,
    get_hyperparameters,
    get_params_for_technique,
)
from experiments.core.modeling.metrics import g_mean_score
from experiments.core.modeling.schema import ExperimentConfig
from experiments.core.modeling.types import ModelType, Technique
from experiments.services.models import ModelVersioningService


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
    """
    Executes a single experiment task.

    Args:
        cfg (ExperimentConfig): Experiment configuration.
        dataset_val (str): The dataset identifier.
        X_mmap_path (str): Path to features memmap.
        y_mmap_path (str): Path to target memmap.
        model_type (ModelType): The model to train.
        technique (Technique): The handling technique.
        seed (int): Random seed.
        checkpoint_path (Path): Path to save the checkpoint.
        model_versioning_service (Optional[ModelVersioningService]): Service to save the model.
    """
    # Reconstruct Dataset enum from identifier
    try:
        dataset = Dataset(dataset_val)  # type: ignore[arg-type]
    except ValueError:
        dataset = Dataset.from_id(str(dataset_val))

    logger.info(
        f"Starting task: {dataset.display_name} | {model_type.display_name} | "
        f"{technique.display_name} | seed={seed}"
    )

    # 1. Checkpoint Check
    if checkpoint_path.exists():
        if cfg.discard_checkpoints:
            logger.info(f"Discarding checkpoint at {checkpoint_path}")
            checkpoint_path.unlink(missing_ok=True)
        else:
            logger.info(f"Checkpoint found at {checkpoint_path}, skipping task.")
            return None

    try:
        # 2. Load Data (Memory Mapped)
        X_mmap = joblib.load(X_mmap_path, mmap_mode="r")
        y_mmap = joblib.load(y_mmap_path, mmap_mode="r")

        # 3. Validation Checks
        _, counts = np.unique(y_mmap, return_counts=True)
        min_class_count = counts.min()

        stratify_y = y_mmap
        internal_cv_folds = cfg.cv_folds

        # Adjust folds if class count is too small
        if min_class_count < 2:
            stratify_y = None
            internal_cv_folds = max(2, min(internal_cv_folds, min_class_count))

        # 4. Split INDICES only
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

        # 5. MATERIALIZE TRAIN SET
        X_train = X_mmap[train_idx]
        y_train = y_mmap[train_idx]

        pipeline = build_pipeline(model_type, technique, seed)
        base_grid = get_hyperparameters(model_type)

        # Retrieve cost grids from Context
        param_grid = get_params_for_technique(model_type, technique, base_grid, cfg.cost_grids)

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

        # 6. Cleanup train set from memory
        del X_train, y_train
        gc.collect()

        # 7. Materialize test set
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
            "dataset": dataset.id,
            "seed": seed,
            "model": model_type.id,
            "technique": technique.id,
            "best_params": str(grid.best_params_),
            "accuracy_balanced": balanced_accuracy_score(y_test, y_pred),
            "g_mean": g_mean_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": auc_score,
        }

        logger.success(
            f"Finished: {dataset.display_name} | {model_type.display_name} | "
            f"{technique.display_name} | seed={seed} -> "
            f"AUC={auc_score:.4f}, F1={metrics['f1_score']:.4f}"
        )

        # Save Model
        if model_versioning_service:
            try:
                version = model_versioning_service.save_model(best_model, None)
                logger.info(f"Saved model {version.id}")
            except Exception as e:
                logger.warning(f"Failed to save model: {e}")

        # 9. Save Checkpoint
        df_res = pd.DataFrame([metrics])
        df_res.to_parquet(checkpoint_path)

        # 10. Cleanup
        del grid, best_model, X_test, y_test, X_mmap, y_mmap
        gc.collect()

        return f"{dataset.id} - {model_type.id} - {technique.id} - Seed {seed}"

    except Exception:
        logger.error(
            f"Failed task {dataset.display_name} {model_type.display_name} "
            f"{technique.display_name} {seed}:\n{traceback.format_exc()}"
        )
        gc.collect()
        return None
