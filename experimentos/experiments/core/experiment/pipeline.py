"""Experiment pipeline orchestrator.

This module provides the ExperimentPipeline class that coordinates
data splitting, model training, evaluation, and persistence for a single experiment.
"""

from dataclasses import dataclass
import gc
import traceback
from typing import Any

from loguru import logger

from experiments.core.experiment.evaluators import ClassificationEvaluator
from experiments.core.experiment.persisters import ParquetExperimentPersister
from experiments.core.experiment.protocols import (
    DataSplitter,
    EstimatorFactory,
    ExperimentContext,
    ExperimentResult,
    ModelEvaluator,
    ModelTrainer,
)
from experiments.core.experiment.splitters import StratifiedDataSplitter
from experiments.core.experiment.trainers import GridSearchTrainer
from experiments.core.modeling.factories import DefaultEstimatorFactory
from experiments.services.model_versioning import ModelVersioningService
from experiments.services.storage import StorageService


@dataclass
class ExperimentPipelineConfig:
    """Configuration for the experiment pipeline.

    Attributes:
        test_size: Fraction of data for testing.
        scoring: Scoring metric for optimization.
        trainer_n_jobs: Parallel jobs for training.
        trainer_verbose: Verbosity for training.
    """

    test_size: float = 0.30
    scoring: str = "roc_auc"
    trainer_n_jobs: int = 1
    trainer_verbose: int = 0


class ExperimentPipeline:
    """Orchestrates a single experiment: Split → Train → Evaluate → Persist.

    This class coordinates the experiment stages using dependency injection
    for maximum flexibility and testability.
    """

    def __init__(
        self,
        splitter: DataSplitter,
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
        persister: ParquetExperimentPersister,
    ) -> None:
        """Initialize the pipeline.

        Args:
            splitter: Component for splitting data.
            trainer: Component for training models.
            evaluator: Component for evaluating models.
            persister: Component for persisting results.
        """
        self._splitter = splitter
        self._trainer = trainer
        self._evaluator = evaluator
        self._persister = persister

    def run(
        self,
        context: ExperimentContext,
    ) -> ExperimentResult:
        """Run a single experiment.

        Args:
            context: The experiment context with configuration.

        Returns:
            ExperimentResult with task_id, metrics, and optionally the model.
        """
        # 1. Check checkpoint
        if self._persister.checkpoint_exists(context.checkpoint_uri):
            if context.discard_checkpoints:
                self._persister.discard_checkpoint(context.checkpoint_uri)
            else:
                logger.info(
                    f"Skipping existing: {context.identity.dataset.id} | "
                    f"seed={context.identity.seed}"
                )
                return ExperimentResult(task_id=None, metrics={})

        logger.info(
            f"Starting: {context.identity.dataset.display_name} | "
            f"{context.identity.model_type.name} | {context.identity.technique.name} | "
            f"seed={context.identity.seed}"
        )

        try:
            # 2. Split data
            data = self._splitter.split(
                context.data.X_path,
                context.data.y_path,
                context.identity.seed,
                context.config.cv_folds,
            )

            if data is None:
                logger.warning(
                    f"Data validation failed for {context.identity.dataset.display_name}"
                )
                return ExperimentResult(task_id=None, metrics={})

            # 3. Train model
            trained = self._trainer.train(
                data,
                context.identity.model_type,
                context.identity.technique,
                context.identity.seed,
                context.config.cv_folds,
                context.config.cost_grids,
            )

            # Free training memory immediately
            X_test, y_test = data.X_test, data.y_test
            del data.X_train, data.y_train
            gc.collect()

            # 4. Evaluate model
            evaluation = self._evaluator.evaluate(trained.estimator, X_test, y_test)

            # 5. Build final metrics with metadata
            metrics: dict[str, Any] = dict(evaluation.metrics)
            metrics.update(
                {
                    "dataset": context.identity.dataset.id,
                    "seed": context.identity.seed,
                    "model": context.identity.model_type.id,
                    "technique": context.identity.technique.id,
                    "best_params": str(trained.best_params),
                }
            )

            logger.success(
                f"Done: {context.identity.dataset.display_name} | "
                f"{context.identity.model_type.name} | seed={context.identity.seed} | "
                f"AUC={metrics['roc_auc']:.4f}"
            )

            # 6. Persist results
            self._persister.save_model(trained.estimator, context)
            self._persister.save_checkpoint(metrics, context.checkpoint_uri)

            task_id = (
                f"{context.identity.dataset.id}-"
                f"{context.identity.model_type.id}-"
                f"{context.identity.seed}"
            )
            return ExperimentResult(
                task_id=task_id,
                metrics=metrics,
                model=trained.estimator,
            )

        except Exception:
            logger.error(f"Failed task: {traceback.format_exc()}")
            return ExperimentResult(task_id=None, metrics={})

        finally:
            gc.collect()


class ExperimentPipelineFactory:
    """Factory for creating pre-configured experiment pipelines."""

    def __init__(
        self,
        storage: StorageService,
        model_versioning_service: ModelVersioningService | None = None,
        estimator_factory: EstimatorFactory | None = None,
    ) -> None:
        """Initialize the factory.

        Args:
            storage: Storage service for file operations.
            model_versioning_service: Optional service for model versioning.
            estimator_factory: Optional factory for creating estimators.
                If not provided, uses DefaultEstimatorFactory.
        """
        self._storage = storage
        self._model_versioning_service = model_versioning_service
        self._estimator_factory = estimator_factory or DefaultEstimatorFactory()

    def create_default_pipeline(
        self,
        config: ExperimentPipelineConfig | None = None,
    ) -> ExperimentPipeline:
        """Create a pipeline with default components.

        Args:
            config: Optional configuration, uses defaults if not provided.

        Returns:
            A configured ExperimentPipeline.
        """
        config = config or ExperimentPipelineConfig()

        return ExperimentPipeline(
            splitter=StratifiedDataSplitter(test_size=config.test_size),
            trainer=GridSearchTrainer(
                estimator_factory=self._estimator_factory,
                scoring=config.scoring,
                n_jobs=config.trainer_n_jobs,
                verbose=config.trainer_verbose,
            ),
            evaluator=ClassificationEvaluator(),
            persister=ParquetExperimentPersister(
                storage=self._storage,
                model_versioning_service=self._model_versioning_service,
            ),
        )

    def create_pipeline(
        self,
        splitter: DataSplitter,
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
        config: ExperimentPipelineConfig | None = None,
    ) -> ExperimentPipeline:
        """Create a pipeline with custom components.

        Args:
            splitter: Custom data splitter.
            trainer: Custom model trainer.
            evaluator: Custom model evaluator.
            config: Optional configuration.

        Returns:
            A configured ExperimentPipeline with custom components.
        """
        return ExperimentPipeline(
            splitter=splitter,
            trainer=trainer,
            evaluator=evaluator,
            persister=ParquetExperimentPersister(
                storage=self._storage,
                model_versioning_service=self._model_versioning_service,
            ),
        )


__all__ = [
    "ExperimentPipelineConfig",
    "ExperimentPipeline",
    "ExperimentPipelineFactory",
]
