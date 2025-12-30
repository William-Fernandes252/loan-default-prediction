"""Experiment pipeline components.

This package provides a dependency-injection based experiment pipeline
for running individual ML experiments with support for:
- Data splitting and validation
- Model training and hyperparameter optimization
- Model evaluation with multiple metrics
- Results and model persistence

Example usage:
    from experiments.core.modeling.experiment import (
        create_experiment_pipeline,
        ExperimentPipelineConfig,
    )

    pipeline = create_experiment_pipeline(storage, config)
    result = pipeline.run(task_context)
"""

from experiments.core.experiment.adapters import (
    ExperimentRunnerFactory,
    create_experiment_runner,
)
from experiments.core.experiment.evaluators import ClassificationEvaluator
from experiments.core.experiment.persisters import (
    CompositeExperimentPersister,
    ParquetExperimentPersister,
)
from experiments.core.experiment.pipeline import (
    ExperimentPipeline,
    ExperimentPipelineConfig,
    create_custom_experiment_pipeline,
    create_experiment_pipeline,
)
from experiments.core.experiment.protocols import (
    DataPaths,
    DataSplitter,
    EvaluationResult,
    ExperimentContext,
    ExperimentIdentity,
    ExperimentPersister,
    ExperimentResult,
    ModelEvaluator,
    ModelTrainer,
    SplitData,
    TrainedModel,
    TrainingConfig,
)
from experiments.core.experiment.splitters import StratifiedDataSplitter
from experiments.core.experiment.trainers import GridSearchTrainer

__all__ = [
    # Protocols
    "DataSplitter",
    "ModelTrainer",
    "ModelEvaluator",
    "ExperimentPersister",
    # Data classes
    "DataPaths",
    "ExperimentIdentity",
    "TrainingConfig",
    "ExperimentContext",
    "SplitData",
    "TrainedModel",
    "EvaluationResult",
    "ExperimentResult",
    "ExperimentPipelineConfig",
    # Implementations
    "StratifiedDataSplitter",
    "GridSearchTrainer",
    "ClassificationEvaluator",
    "ParquetExperimentPersister",
    "CompositeExperimentPersister",
    # Adapters
    "create_experiment_runner",
    "ExperimentRunnerFactory",
    # Pipeline
    "ExperimentPipeline",
    "create_experiment_pipeline",
    "create_custom_experiment_pipeline",
]
