from unittest.mock import MagicMock

import polars as pl
import pytest

from experiments.core.data.datasets import Dataset
from experiments.core.modeling.classifiers import ModelType, Technique
from experiments.core.predictions.repository import RawPredictions
from experiments.core.training.data import TrainingData
from experiments.core.training.splitters import SplitData
from experiments.core.training.trainers import TrainedModel
from experiments.lib.pipelines import TaskStatus
from experiments.pipelines.training.factory import (
    TrainingPipelineFactory,
    load_training_data,
    predict,
    split_data,
    train_model,
)
from experiments.pipelines.training.pipeline import (
    TrainingPipelineContext,
    TrainingPipelineState,
)


class DescribeLoadTrainingData:
    @pytest.fixture
    def mock_loader(self) -> MagicMock:
        loader = MagicMock()
        loader.load_training_data.return_value = TrainingData(
            X=pl.DataFrame({"feature": [1, 2, 3]}),  # type: ignore[arg-type]
            y=pl.DataFrame({"target": [0, 1, 0]}),  # type: ignore[arg-type]
        )
        return loader

    @pytest.fixture
    def context(self, mock_loader: MagicMock) -> TrainingPipelineContext:
        return TrainingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.SMOTE,
            classifier_factory=MagicMock(),
            training_data_loader=mock_loader,
            trainer=MagicMock(),
            data_splitter=MagicMock(),
            seed=42,
        )

    def it_loads_training_data_from_loader(
        self, context: TrainingPipelineContext, mock_loader: MagicMock
    ) -> None:
        state: TrainingPipelineState = {}

        result = load_training_data(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_loader.load_training_data.assert_called_once_with(Dataset.TAIWAN_CREDIT)

    def it_updates_state_with_training_data(self, context: TrainingPipelineContext) -> None:
        state: TrainingPipelineState = {}

        result = load_training_data(state, context)

        assert "training_data" in result.state
        assert isinstance(result.state["training_data"], TrainingData)

    def it_returns_success_message(self, context: TrainingPipelineContext) -> None:
        state: TrainingPipelineState = {}

        result = load_training_data(state, context)

        assert result.message == "Loaded training data."


class DescribeSplitData:
    @pytest.fixture
    def mock_splitter(self) -> MagicMock:
        splitter = MagicMock()
        splitter.split.return_value = SplitData(
            X_train=pl.DataFrame({"feature": [1, 2]}),  # type: ignore[arg-type]
            X_test=pl.DataFrame({"feature": [3]}),  # type: ignore[arg-type]
            y_train=pl.DataFrame({"target": [0, 1]}),  # type: ignore[arg-type]
            y_test=pl.DataFrame({"target": [0]}),  # type: ignore[arg-type]
        )
        return splitter

    @pytest.fixture
    def context(self, mock_splitter: MagicMock) -> TrainingPipelineContext:
        return TrainingPipelineContext(
            dataset=Dataset.LENDING_CLUB,
            model_type=ModelType.SVM,
            technique=Technique.BASELINE,
            classifier_factory=MagicMock(),
            training_data_loader=MagicMock(),
            trainer=MagicMock(),
            data_splitter=mock_splitter,
            seed=123,
        )

    def it_splits_data_using_splitter(
        self, context: TrainingPipelineContext, mock_splitter: MagicMock
    ) -> None:
        training_data = TrainingData(
            X=pl.DataFrame({"feature": [1, 2, 3]}),  # type: ignore[arg-type]
            y=pl.DataFrame({"target": [0, 1, 0]}),  # type: ignore[arg-type]
        )
        state: TrainingPipelineState = {"training_data": training_data}

        result = split_data(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_splitter.split.assert_called_once_with(data=training_data, seed=123)

    def it_updates_state_with_split_data(self, context: TrainingPipelineContext) -> None:
        state: TrainingPipelineState = {
            "training_data": TrainingData(
                X=pl.DataFrame({"feature": [1]}),  # type: ignore[arg-type]
                y=pl.DataFrame({"target": [0]}),  # type: ignore[arg-type]
            )
        }

        result = split_data(state, context)

        assert "data_split" in result.state
        assert isinstance(result.state["data_split"], SplitData)

    def it_returns_failure_when_no_training_data(self, context: TrainingPipelineContext) -> None:
        state: TrainingPipelineState = {}

        result = split_data(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "Training data not loaded" in result.message


class DescribeTrainModel:
    @pytest.fixture
    def mock_factory(self) -> MagicMock:
        factory = MagicMock()
        factory.create_model.return_value = MagicMock()  # Mock classifier
        return factory

    @pytest.fixture
    def mock_trainer(self) -> MagicMock:
        trainer = MagicMock()
        mock_model = MagicMock()
        trainer.train.return_value = TrainedModel(
            model=mock_model,
            params={"max_depth": 10},
            seed=42,
        )
        return trainer

    @pytest.fixture
    def context(self, mock_factory: MagicMock, mock_trainer: MagicMock) -> TrainingPipelineContext:
        return TrainingPipelineContext(
            dataset=Dataset.CORPORATE_CREDIT_RATING,
            model_type=ModelType.XGBOOST,
            technique=Technique.SMOTE_TOMEK,
            classifier_factory=mock_factory,
            training_data_loader=MagicMock(),
            trainer=mock_trainer,
            data_splitter=MagicMock(),
            seed=99,
            use_gpu=True,
            n_jobs=4,
        )

    def it_creates_classifier_from_factory(
        self, context: TrainingPipelineContext, mock_factory: MagicMock
    ) -> None:
        state: TrainingPipelineState = {
            "data_split": SplitData(
                X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
                X_test=pl.DataFrame({"f": [2]}),  # type: ignore[arg-type]
                y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
                y_test=pl.DataFrame({"t": [1]}),  # type: ignore[arg-type]
            )
        }

        result = train_model(state, context)

        mock_factory.create_model.assert_called_once_with(
            model_type=ModelType.XGBOOST,
            technique=Technique.SMOTE_TOMEK,
            seed=99,
            use_gpu=True,
            n_jobs=4,
        )
        assert result.status == TaskStatus.SUCCESS

    def it_trains_model_with_trainer(
        self, context: TrainingPipelineContext, mock_trainer: MagicMock
    ) -> None:
        data_split = SplitData(
            X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
            X_test=pl.DataFrame({"f": [2]}),  # type: ignore[arg-type]
            y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
            y_test=pl.DataFrame({"t": [1]}),  # type: ignore[arg-type]
        )
        state: TrainingPipelineState = {"data_split": data_split}

        result = train_model(state, context)

        assert mock_trainer.train.called
        assert result.status == TaskStatus.SUCCESS

    def it_updates_state_with_trained_model(self, context: TrainingPipelineContext) -> None:
        state: TrainingPipelineState = {
            "data_split": SplitData(
                X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
                X_test=pl.DataFrame({"f": [2]}),  # type: ignore[arg-type]
                y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
                y_test=pl.DataFrame({"t": [1]}),  # type: ignore[arg-type]
            )
        }

        result = train_model(state, context)

        assert "trained_model" in result.state
        assert isinstance(result.state["trained_model"], TrainedModel)

    def it_returns_failure_when_no_data_split(self, context: TrainingPipelineContext) -> None:
        state: TrainingPipelineState = {}

        result = train_model(state, context)

        assert result.status == TaskStatus.FAILURE
        assert result.message is not None and "Data not split" in result.message


class DescribePredict:
    @pytest.fixture
    def context(self) -> TrainingPipelineContext:
        return TrainingPipelineContext(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            classifier_factory=MagicMock(),
            training_data_loader=MagicMock(),
            trainer=MagicMock(),
            data_splitter=MagicMock(),
            seed=42,
        )

    def it_makes_predictions_on_test_data(self, context: TrainingPipelineContext) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = pl.Series([0, 1, 0])

        X_test = pl.DataFrame({"feature": [1, 2, 3]})
        y_test = pl.DataFrame({"target": [0, 1, 0]})

        state: TrainingPipelineState = {
            "trained_model": TrainedModel(
                model=mock_model,
                params={"max_depth": 5},
                seed=42,
            ),
            "data_split": SplitData(
                X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
                X_test=X_test,  # type: ignore[arg-type]
                y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
                y_test=y_test,  # type: ignore[arg-type]
            ),
        }

        result = predict(state, context)

        assert result.status == TaskStatus.SUCCESS
        mock_model.predict.assert_called_once()
        # Verify the prediction was called with X_test
        assert mock_model.predict.call_args[0][0].equals(X_test)

    def it_updates_state_with_predictions(self, context: TrainingPipelineContext) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = pl.Series([0, 1])

        state: TrainingPipelineState = {
            "trained_model": TrainedModel(
                model=mock_model,
                params={"C": 1.0},
                seed=123,
            ),
            "data_split": SplitData(
                X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
                X_test=pl.DataFrame({"f": [2, 3]}),  # type: ignore[arg-type]
                y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
                y_test=pl.DataFrame({"t": [1, 0]}),  # type: ignore[arg-type]
            ),
        }

        result = predict(state, context)

        assert "predictions" in result.state
        assert isinstance(result.state["predictions"], RawPredictions)

    def it_returns_success_message(self, context: TrainingPipelineContext) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = pl.Series([1])

        state: TrainingPipelineState = {
            "trained_model": TrainedModel(
                model=mock_model,
                params={"n_estimators": 100},
                seed=42,
            ),
            "data_split": SplitData(
                X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
                X_test=pl.DataFrame({"f": [2]}),  # type: ignore[arg-type]
                y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
                y_test=pl.DataFrame({"t": [1]}),  # type: ignore[arg-type]
            ),
        }

        result = predict(state, context)

        assert result.message == "Predictions made successfully."

    def it_keeps_trained_model_in_state(self, context: TrainingPipelineContext) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = pl.Series([0, 1])

        trained_model = TrainedModel(
            model=mock_model,
            params={"n_estimators": 100},
            seed=42,
        )
        state: TrainingPipelineState = {
            "trained_model": trained_model,
            "data_split": SplitData(
                X_train=pl.DataFrame({"f": [1]}),  # type: ignore[arg-type]
                X_test=pl.DataFrame({"f": [2, 3]}),  # type: ignore[arg-type]
                y_train=pl.DataFrame({"t": [0]}),  # type: ignore[arg-type]
                y_test=pl.DataFrame({"t": [1, 0]}),  # type: ignore[arg-type]
            ),
        }

        result = predict(state, context)

        assert result.state.get("trained_model") == trained_model


class DescribeTrainingPipelineFactory:
    @pytest.fixture
    def factory(self) -> TrainingPipelineFactory:
        return TrainingPipelineFactory()

    def it_creates_pipeline_with_correct_name(self, factory: TrainingPipelineFactory) -> None:
        pipeline = factory.create_pipeline("test_training_pipeline")

        assert pipeline.name == "test_training_pipeline"

    def it_creates_pipeline_with_all_steps(self, factory: TrainingPipelineFactory) -> None:
        pipeline = factory.create_pipeline()

        step_names = [step.step.name for step in pipeline.steps]
        assert "LoadTrainingData" in step_names
        assert "SplitData" in step_names
        assert "TrainModel" in step_names
        assert "Predict" in step_names

    def it_creates_pipeline_with_steps_in_order(self, factory: TrainingPipelineFactory) -> None:
        pipeline = factory.create_pipeline()

        step_names = [step.step.name for step in pipeline.steps]
        assert step_names == ["LoadTrainingData", "SplitData", "TrainModel", "Predict"]

    def it_creates_pipeline_without_conditional_steps(
        self, factory: TrainingPipelineFactory
    ) -> None:
        pipeline = factory.create_pipeline()

        conditional_steps = [step for step in pipeline.steps if step.condition is not None]
        assert len(conditional_steps) == 0
