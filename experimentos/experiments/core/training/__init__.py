"""Training pipeline components.

This package provides a dependency-injection based training pipeline
for running machine learning experiments with support for:
- Task generation from dataset/model/technique combinations
- Sequential and parallel execution
- Checkpoint-based result persistence
- Results consolidation

Example usage:
    from experiments.core.training import (
        TrainingPipelineFactory,
        TrainingPipelineConfig,
    )

    factory = TrainingPipelineFactory(
        data_provider=data_manager,
        checkpoint_provider=context,
        versioning_provider=model_service,
        experiment_runner=run_experiment,
    )

    config = TrainingPipelineConfig(
        cv_folds=5,
        cost_grids=COST_GRIDS,
        num_seeds=30,
    )

    pipeline = factory.create_parallel_pipeline(config, n_jobs=-1)
    results = pipeline.run_all(datasets)
"""

from experiments.core.training.executors import (
    BaseExecutor,
    ParallelExecutor,
    SequentialExecutor,
)
from experiments.core.training.generators import (
    ExperimentTaskGenerator,
    TaskGeneratorConfig,
)
from experiments.core.training.persisters import (
    ConsolidationPathProvider,
    ParquetCheckpointPersister,
)
from experiments.core.training.pipeline import (
    TrainingPipeline,
    TrainingPipelineConfig,
    TrainingPipelineFactory,
)
from experiments.core.training.protocols import (
    CheckpointPathProvider,
    DataProvider,
    ExperimentRunner,
    ExperimentTask,
    ModelVersioningProvider,
    ResultsConsolidator,
    TaskGenerator,
    TrainingExecutor,
)

__all__ = [
    # Protocols
    "TaskGenerator",
    "CheckpointPathProvider",
    "ModelVersioningProvider",
    "DataProvider",
    "TrainingExecutor",
    "ResultsConsolidator",
    "ExperimentRunner",
    "ConsolidationPathProvider",
    # Data classes
    "ExperimentTask",
    "TaskGeneratorConfig",
    "TrainingPipelineConfig",
    # Implementations
    "ExperimentTaskGenerator",
    "BaseExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
    "ParquetCheckpointPersister",
    # Pipeline
    "TrainingPipeline",
    "TrainingPipelineFactory",
]
