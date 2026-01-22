"""Factory for creating prediction pipelines.

Currently, the prediction pipeline is used only to test the trained models with the test data. In the future, it may be extended to include actual inference capabilities for external data.
"""

from experiments.lib.pipelines import Pipeline, TaskResult, TaskStatus
from experiments.pipelines.predictions.pipeline import (
    PredictionsPipeline,
    PredictionsPipelineContext,
    PredictionsPipelineState,
    PredictionsPipelineTaskResult,
)


def retrieve_trained_model(
    state: PredictionsPipelineState, context: PredictionsPipelineContext
) -> PredictionsPipelineTaskResult:
    """Retrieve the trained model into the pipeline state.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with trained model retrieved.

    Raises:
        ValueError: If no trained model is found for the given identifier and dataset.
    """
    trained_model = context.trained_model_loader.load_model(context.dataset, context.model_id)
    state["trained_model"] = trained_model
    return TaskResult(state, TaskStatus.SUCCESS, "Trained model retrieved successfully.")


def load_training_data(
    state: PredictionsPipelineState, context: PredictionsPipelineContext
) -> PredictionsPipelineTaskResult:
    """Load training data into the pipeline state.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with training data loaded.

    Raises:
        ValueError: If the training data cannot be loaded for the given dataset.
    """
    training_data = context.training_data_loader.load_training_data(context.dataset)
    state["training_data"] = training_data
    return TaskResult(state, TaskStatus.SUCCESS, "Training data loaded successfully.")


def split_data(
    state: PredictionsPipelineState, context: PredictionsPipelineContext
) -> PredictionsPipelineTaskResult:
    """Split the training data and store the result in the pipeline state.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with data split information.

    Raises:
        ValueError: If the data splitting fails.
    """
    split_data = context.data_splitter.split(state["training_data"], seed=context.seed)
    state["data_split"] = split_data
    return TaskResult(state, TaskStatus.SUCCESS, "Data split successfully.")


def predict(
    state: PredictionsPipelineState, context: PredictionsPipelineContext
) -> PredictionsPipelineTaskResult:
    """Make predictions using the trained model.

    Args:
        state: Current pipeline state.
        context: Current pipeline context.

    Returns:
        Updated pipeline state with prediction results.
    """
    predictions = state["trained_model"].model.predict(state["data_split"].X_test)
    state["predictions"] = predictions
    return TaskResult(state, TaskStatus.SUCCESS, "Predictions made successfully.")


class PredictionsPipelineFactory:
    """Factory for creating prediction pipelines."""

    def create_pipeline(self, name: str = "PredictionsPipeline") -> PredictionsPipeline:
        """Create a predictions pipeline.

        Returns:
            A PredictionsPipeline instance.
        """

        pipeline = Pipeline[PredictionsPipelineState, PredictionsPipelineContext](name=name)

        pipeline.add_step("RetrieveTrainedModel", retrieve_trained_model)
        pipeline.add_step("LoadTrainingData", load_training_data)
        pipeline.add_step("SplitData", split_data)
        pipeline.add_step("Predict", predict)

        return pipeline
