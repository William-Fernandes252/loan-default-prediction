"""Defines lifecycle hooks functionalities for pipeline execution."""

import enum
from typing import TYPE_CHECKING, Generic, Protocol

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.pipeline import Pipeline
from experiments.lib.pipelines.state import State
from experiments.lib.pipelines.tasks import TaskResult

if TYPE_CHECKING:
    from experiments.lib.pipelines.execution import PipelineExecutionResult


class ErrorAction(enum.IntEnum):
    """Actions to take when an error occurs during pipeline execution.

    - `ABORT`: Stop the pipeline execution immediately.
    - `RETRY`: Retry the failed step.
    - `IGNORE`: Ignore the failed step and continue with the next one.
    - `PANIC`: Raise the error immediately without any handling.
    """

    IGNORE = enum.auto()
    RETRY = enum.auto()
    ABORT = enum.auto()
    PANIC = enum.auto()


class UserAction(enum.IntEnum):
    """Actions to take when user intervention is required during pipeline execution.

    - `ABORT`: Stop the pipeline execution immediately.
    - `PROCEED`: Continue with the next step.
    """

    PROCEED = enum.auto()
    RETRY = enum.auto()
    ABORT = enum.auto()


class PipelineObserver(Generic[State, Context], Protocol):
    def on_step_start(
        self, pipeline: Pipeline[State, Context], step_name: str, current_state: State
    ) -> None:
        """Called when a pipeline step starts.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step that is starting.
            current_state: The current state before the step execution.
        """
        ...

    def on_step_finish(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        result: TaskResult[State],
    ) -> None:
        """Called when a pipeline step finishes.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step that has finished.
            result: The result of the step execution.
        """
        ...

    def on_error(
        self, pipeline: Pipeline[State, Context], step_name: str, error: Exception
    ) -> ErrorAction:
        """Called when an error occurs during a pipeline step.

        It allows the observer to decide how to handle the error by returning an `ErrorAction`.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step where the error occurred.
            error: The exception that was raised.
        """
        ...

    def on_action_required(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        message: str,
    ) -> ErrorAction:
        """Called when a pipeline step requires user action to proceed.

        It allows the observer to decide how to handle the situation by returning an `ErrorAction`.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step that requires action.
            message: A message describing the required action.
        """
        ...

    def on_pipeline_start(self, pipeline: Pipeline[State, Context]) -> None:
        """Called when the pipeline execution starts.

        Args:
            pipeline: The pipeline that is starting.
        """
        ...

    def on_pipeline_finish(
        self,
        pipeline: Pipeline[State, Context],
        result: "PipelineExecutionResult[State, Context]",
    ) -> None:
        """Called when the pipeline execution finishes.

        Args:
            pipeline: The pipeline that has finished.
            result: The result of the pipeline execution.
        """
        ...


class IgnoreActionsPipelineObserver(Generic[State, Context], PipelineObserver[State, Context]):
    """A pipeline observer that ignores all events. Useful to logging or monitoring."""

    def on_step_start(
        self, pipeline: Pipeline[State, Context], step_name: str, current_state: State
    ) -> None: ...

    def on_step_finish(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        result: TaskResult[State],
    ) -> None: ...

    def on_error(
        self, pipeline: Pipeline[State, Context], step_name: str, error: Exception
    ) -> ErrorAction:
        return ErrorAction.IGNORE

    def on_action_required(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        message: str,
    ) -> ErrorAction:
        return ErrorAction.IGNORE

    def on_pipeline_start(self, pipeline: Pipeline[State, Context]) -> None: ...

    def on_pipeline_finish(
        self,
        pipeline: Pipeline[State, Context],
        result: "PipelineExecutionResult[State, Context]",
    ) -> None: ...


class PipelineActionRequestHandler(Generic[State, Context], Protocol):
    """An interface for handling user action requests during pipeline execution."""

    def __call__(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        step_result: TaskResult[State],
    ) -> UserAction:
        """Handle a user action request during pipeline execution.

        Args:
            pipeline: The pipeline where the action is requested.
            step_name: The name of the step requesting action.
            message: A message describing the required action.

        Returns:
            UserAction: The action to take in response to the request.
        """
        ...
