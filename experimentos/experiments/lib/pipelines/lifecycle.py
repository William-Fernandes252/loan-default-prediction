"""Defines lifecycle hooks functionalities for pipeline execution."""

import enum
from typing import TYPE_CHECKING, Protocol

from experiments.lib.pipelines.pipeline import Pipeline
from experiments.lib.pipelines.tasks import TaskResult

if TYPE_CHECKING:
    from experiments.lib.pipelines.execution import PipelineExecutionResult


class Action(enum.IntEnum):
    """Actions observers can request during pipeline execution.

    Actions are ordered by severity, allowing the executor to aggregate
    responses from multiple observers by taking the maximum value.

    Semantics vary by lifecycle hook:

    - `PROCEED`: Continue normally.
        - on_step_start: Execute the step
        - on_step_finish: Continue to next step
        - on_error: Skip the step and continue
        - on_pipeline_start: Start execution
        - on_pipeline_finish: Complete normally

    - `RETRY`: Retry the current unit.
        - on_step_start: Not applicable (treated as PROCEED)
        - on_step_finish: Re-execute this step
        - on_error: Retry the failed step
        - on_pipeline_start: Not applicable (treated as PROCEED)
        - on_pipeline_finish: Re-execute the entire pipeline from the beginning

    - `ABORT`: Stop this pipeline, but let other pipelines continue.
        - All hooks: Immediately stop this pipeline's execution

    - `PANIC`: Stop all pipeline execution immediately.
        - All hooks: Shut down the executor and raise the error (if any)
    """

    PROCEED = enum.auto()
    RETRY = enum.auto()
    ABORT = enum.auto()
    PANIC = enum.auto()


class PipelineObserver[State, Context](Protocol):
    """Observer protocol for pipeline execution lifecycle events.

    Observers receive notifications about pipeline and step lifecycle events,
    and can influence execution flow through their return values.

    All observer methods must be thread-safe, as they may be called from
    multiple worker threads concurrently when executing multiple pipelines.

    All methods return an `Action` that controls pipeline execution flow.
    When multiple observers are registered, the executor takes the action
    with the highest severity (max value).
    """

    def on_step_start(
        self, pipeline: Pipeline[State, Context], step_name: str, current_state: State
    ) -> Action:
        """Called when a pipeline step is about to start.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step that is starting.
            current_state: The current state before the step execution.

        Returns:
            Action: PROCEED to execute, ABORT to skip and stop pipeline,
                PANIC to stop all pipelines.
        """
        ...

    def on_step_finish(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        result: TaskResult[State],
    ) -> Action:
        """Called when a pipeline step finishes successfully.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step that has finished.
            result: The result of the step execution.

        Returns:
            Action: PROCEED to continue, RETRY to re-execute this step,
                ABORT to stop pipeline, PANIC to stop all pipelines.
        """
        ...

    def on_step_skipped(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        reason: str,
    ) -> Action:
        """Called when a pipeline step is skipped.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step that was skipped.
            reason: The reason why the step was skipped.

        Returns:
            Action: PROCEED to continue, ABORT to stop pipeline,
                PANIC to stop all pipelines.
        """
        ...

    def on_error(
        self, pipeline: Pipeline[State, Context], step_name: str, error: Exception
    ) -> Action:
        """Called when an error occurs during a pipeline step.

        Args:
            pipeline: The pipeline to which the step belongs.
            step_name: The name of the step where the error occurred.
            error: The exception that was raised.

        Returns:
            Action: PROCEED to skip step and continue, RETRY to retry,
                ABORT to stop pipeline, PANIC to stop all and raise error.
        """
        ...

    def on_pipeline_start(self, pipeline: Pipeline[State, Context]) -> Action:
        """Called when the pipeline execution starts.

        Args:
            pipeline: The pipeline that is starting.

        Returns:
            Action: `PROCEED` to start, `ABORT` to skip this pipeline,
                `PANIC` to stop all pipelines.
        """
        ...

    def on_pipeline_finish(
        self,
        pipeline: Pipeline[State, Context],
        result: "PipelineExecutionResult[State, Context]",
    ) -> Action:
        """Called when the pipeline execution finishes.

        Args:
            pipeline: The pipeline that has finished.
            result: The result of the pipeline execution.

        Returns:
            Action: `PROCEED` to complete, `RETRY` to re-execute entire pipeline,
                `ABORT` to mark as aborted, `PANIC` to stop all pipelines.
        """
        ...


class IgnoreAllObserver[State, Context]:
    """A pipeline observer that proceeds through all events.

    This observer returns `PROCEED` for all events, allowing pipelines to
    continue normally. Useful as a base class for logging observers or
    when combined with other observers.
    """

    def on_step_start(
        self, pipeline: Pipeline[State, Context], step_name: str, current_state: State
    ) -> Action:
        return Action.PROCEED

    def on_step_skipped(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        reason: str,
    ) -> Action:
        return Action.PROCEED

    def on_step_finish(
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        result: TaskResult[State],
    ) -> Action:
        return Action.PROCEED

    def on_error(
        self, pipeline: Pipeline[State, Context], step_name: str, error: Exception
    ) -> Action:
        return Action.PROCEED

    def on_pipeline_start(self, pipeline: Pipeline[State, Context]) -> Action:
        return Action.PROCEED

    def on_pipeline_finish(
        self,
        pipeline: Pipeline[State, Context],
        result: "PipelineExecutionResult[State, Context]",
    ) -> Action:
        return Action.PROCEED


class AbortOnErrorObserver[State, Context](IgnoreAllObserver[State, Context]):
    """A pipeline observer that aborts the pipeline on any error.

    This observer returns `ABORT` for errors and `PROCEED` for other events,
    providing a fail-fast behavior for individual pipelines while
    allowing other pipelines to continue.
    """

    def on_error(
        self, pipeline: Pipeline[State, Context], step_name: str, error: Exception
    ) -> Action:
        return Action.ABORT
