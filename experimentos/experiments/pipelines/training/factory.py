"""Definition of the training pipeline factory."""

from experiments.core.predictions.repository import RawPredictions
from experiments.core.training.trainers import ModelTrainRequest
from experiments.lib.pipelines import Pipeline, TaskResult, TaskStatus
from experiments.pipelines.training.pipeline import (
    TrainingPipeline,
    TrainingPipelineContext,
    TrainingPipelineState,
    TrainingPipelineTaskResult,
)


def load_training_data(
    state: TrainingPipelineState, context: TrainingPipelineContext
) -> TrainingPipelineTaskResult:
    """Load training data into the pipeline state.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with training data loaded.
    """
    training_data = context.training_data_loader.load_training_data(context.dataset)
    state["training_data"] = training_data
    return TaskResult(state, TaskStatus.SUCCESS, "Loaded training data.")


def split_data(
    state: TrainingPipelineState, context: TrainingPipelineContext
) -> TrainingPipelineTaskResult:
    """Split data into training and testing sets.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with split data.
    """
    if "training_data" not in state:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "Training data not loaded; cannot split data.",
        )

    state["data_split"] = context.data_splitter.split(
        data=state["training_data"],
        seed=context.seed,
    )
    # Free training data immediately after split to reduce memory usage
    del state["training_data"]
    return TaskResult(state, TaskStatus.SUCCESS, "Data split successfully.")


def train_model(
    state: TrainingPipelineState, context: TrainingPipelineContext
) -> TrainingPipelineTaskResult:
    """Train the model using the provided trainer.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with trained model.
    """
    if "data_split" not in state:
        return TaskResult(
            state,
            TaskStatus.FAILURE,
            "Data not split; cannot train model.",
        )

    classifier = context.classifier_factory.create_model(
        model_type=context.model_type,
        technique=context.technique,
        seed=context.seed,
        use_gpu=context.use_gpu,
        n_jobs=context.n_jobs,
    )

    trained_model = context.trainer.train(
        request=ModelTrainRequest(
            classifier=classifier,
            model_type=context.model_type,
            technique=context.technique,
            data=state["data_split"],
            seed=context.seed,
        ),
    )

    state["trained_model"] = trained_model
    return TaskResult(state, TaskStatus.SUCCESS, "Model trained successfully.")


def predict(
    state: TrainingPipelineState, context: TrainingPipelineContext
) -> TrainingPipelineTaskResult:
    """Make predictions using the trained model.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with prediction results.
    """
    predictions = state["trained_model"].model.predict(state["data_split"].X_test)
    state["predictions"] = RawPredictions(
        target=state["data_split"].y_test,
        prediction=predictions,
    )
    # Free split data immediately after predictions to reduce memory usage
    # Keep `trained_model` so callers like TrainingExecutor can return it.
    del state["data_split"]
    return TaskResult(state, TaskStatus.SUCCESS, "Predictions made successfully.")


class TrainingPipelineFactory:
    """Factory for creating training pipelines based on model type and technique.

    A training pipeline consists of steps to load data, split it, and train a model.
    """

    def create_pipeline(
        self,
        name: str = "TrainingPipeline",
    ) -> TrainingPipeline:
        """Create a training pipeline.

        Returns:
            TrainingPipeline: The constructed training pipeline.
        """
        pipeline = Pipeline[TrainingPipelineState, TrainingPipelineContext](name=name)

        pipeline.add_step("LoadTrainingData", load_training_data)
        pipeline.add_step("SplitData", split_data)
        pipeline.add_step("TrainModel", train_model)
        pipeline.add_step("Predict", predict)

        return pipeline
