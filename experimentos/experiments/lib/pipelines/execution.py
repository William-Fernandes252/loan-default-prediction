"""Defines the pipeline execution engine."""

from collections.abc import Set
from dataclasses import dataclass
from queue import Queue
import time
from typing import Annotated, Any, Generic, final

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.lifecycle import (
    ErrorAction,
    PipelineActionRequestHandler,
    PipelineObserver,
    UserAction,
)
from experiments.lib.pipelines.pipeline import Pipeline
from experiments.lib.pipelines.state import State
from experiments.lib.pipelines.steps import Step
from experiments.lib.pipelines.tasks import TaskResult, TaskStatus


@dataclass
class StepTrace:
    """Trace information for a single pipeline step execution."""

    name: Annotated[str, "The name of the step."]
    duration: Annotated[float, "Duration of the step execution in seconds."]
    status: Annotated[TaskStatus, "The status of the step execution."]
    message: Annotated[str | None, "The message returned from the step execution, if any."]
    error: Annotated[Exception | None, "The exception raised during step execution, if any."]


@dataclass(frozen=True, slots=True)
class PipelineExecutionResult(Generic[State, Context]):
    """Result of pipeline execution."""

    step_traces: Annotated[list[StepTrace], "List of step execution traces."]
    context: Annotated[Context, "The context used during pipeline execution."]
    final_state: Annotated[State, "Final state after pipeline execution."]

    @property
    def last_executed_step(self) -> str | None:
        """Get the name of the last executed step, if any."""
        if not self.step_traces:
            return None
        return self.step_traces[-1].name

    @property
    def errors(self) -> dict[str, Exception]:
        """Get a mapping of step names to errors encountered during execution."""
        return {trace.name: trace.error for trace in self.step_traces if trace.error is not None}

    @property
    def durations(self) -> dict[str, float]:
        """Get a mapping of step names to their execution durations."""
        return {trace.name: trace.duration for trace in self.step_traces}

    def total_duration(self) -> float:
        """Calculate the total duration of the pipeline execution."""
        return sum(self.durations.values())

    def succeeded(self) -> bool:
        """Check if the pipeline execution succeeded without errors."""
        return len(self.errors) == 0

    def last_error(self) -> Exception | None:
        """Get the last error that occurred during pipeline execution, if any."""
        if not self.errors:
            return None
        last_step = self.last_executed_step
        if last_step and last_step in self.errors:
            return self.errors[last_step]
        return None


@final
class PipelineExecutor:
    """Orchestrates the execution of pipeline steps.

    It manages the sequential execution of steps in a pipeline, handles errors using provided error handlers, and collects execution results.

    Args:
        default_observers: A set of default observers to notify during execution. These are used in addition to any observers provided at execution time.
        default_error_action: The default action to take on errors if no specific handler is provided. Default is `ErrorAction.PANIC`, i.e., raise the error immediately without any handling.
    """

    def __init__(
        self,
        default_observers: Set[PipelineObserver[Any, Any]] | None = None,
        default_error_action: ErrorAction = ErrorAction.PANIC,
        default_action_request_handler: PipelineActionRequestHandler[Any, Any] | None = None,
    ) -> None:
        self._default_observers = default_observers or set()
        self._default_error_action = default_error_action
        self._default_action_request_handler = default_action_request_handler

    def execute[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        initial_state: State,
        observers: Set[PipelineObserver[State, Context]] | None = None,
        action_request_handler: PipelineActionRequestHandler[State, Context] | None = None,
    ) -> PipelineExecutionResult[State, Context]:
        """Execute a pipeline from the initial state.

        Args:
            pipeline: The pipeline to execute.
            initial_state: The initial state to start the pipeline with.
            observers: Optional set of observers to notify during execution.

        Returns:
            PipelineExecutionResult: The result of the pipeline execution.
        """
        self.__notify_pipeline_start(pipeline, observers)

        context = pipeline.context
        step_traces: list[StepTrace] = []
        steps = self.__create_steps_queue(pipeline)
        current_state = initial_state
        while not steps.empty():
            step = steps.get()
            step_result: TaskResult[State] | None = None
            step_start_time = time.time()
            self.__notify_step_start(pipeline, step.name, current_state, observers)
            try:
                step_result = step(current_state, context)
                if step_result.status == TaskStatus.REQUIRES_ACTION and (
                    handler := self.__get_action_request_handler(action_request_handler)
                ):
                    action = handler(pipeline, step.name, step_result)
                    if action == UserAction.ABORT:
                        break
                    elif action == UserAction.RETRY:
                        steps.put(step)
                        continue
                current_state = step_result.state
            except Exception as e:
                action = self.__notify_error(pipeline, step.name, e, observers)
                if action == ErrorAction.IGNORE:
                    continue
                elif action == ErrorAction.RETRY:
                    steps.put(step)
                elif action == ErrorAction.ABORT:
                    break
                elif action == ErrorAction.PANIC:
                    raise e
            else:
                self.__notify_step_finish(pipeline, step.name, step_result, observers)
            finally:
                end_time = time.time()
                duration = end_time - step_start_time
                step_traces.append(
                    StepTrace(
                        name=step.name,
                        duration=duration,
                        status=step_result.status if step_result else TaskStatus.ERROR,
                        message=step_result.message if step_result else None,
                        error=step_result.error if step_result else None,
                    )
                )

        result = PipelineExecutionResult(
            step_traces=step_traces,
            context=context,
            final_state=current_state,
        )
        self.__notify_pipeline_finish(
            pipeline,
            result,
            observers,
        )

        return result

    @staticmethod
    def __create_steps_queue(
        pipeline: Pipeline[State, Context],
    ) -> Queue[Step[State, Context]]:
        """Create a queue of steps from the pipeline."""
        steps_queue: Queue[Step[State, Context]] = Queue()
        for step in pipeline.steps:
            steps_queue.put(step)
        return steps_queue

    def __notify_pipeline_start[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> None:
        """Notify observers that the pipeline is starting."""
        all_observers = self.__merge_observers(observers)
        for observer in all_observers:
            observer.on_pipeline_start(pipeline)

    def __notify_pipeline_finish[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        result: PipelineExecutionResult[State, Context],
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> None:
        """Notify observers that the pipeline has ended."""
        all_observers = self.__merge_observers(observers)
        for observer in all_observers:
            observer.on_pipeline_finish(pipeline, result)

    def __notify_step_start[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        current_state: State,
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> None:
        """Notify observers that a step is starting."""
        all_observers = self.__merge_observers(observers)
        for observer in all_observers:
            observer.on_step_start(pipeline, step_name, current_state)

    def __notify_step_finish[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        step_result: TaskResult[State],
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> None:
        """Notify observers that a step has ended."""
        all_observers = self.__merge_observers(observers)
        for observer in all_observers:
            observer.on_step_finish(pipeline, step_name, step_result)

    def __notify_error[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        step_name: str,
        error: Exception,
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> ErrorAction:
        """Notify observers that an error has occurred.

        Returns:
            ErrorAction: The aggregated error action from all observers. It is determined by taking the maximum severity action returned by each observer (`IGNORE` < `SKIP` < `RETRY` < `ABORT` < `PANIC`).
        """
        all_observers = self.__merge_observers(observers)
        if len(all_observers) == 0:
            return self._default_error_action
        return ErrorAction(
            max(
                [observer.on_error(pipeline, step_name, error).value for observer in all_observers]
            )
        )

    def __merge_observers(
        self,
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> Set[PipelineObserver[State, Context]]:
        """Merge default observers with provided observers."""
        return self._default_observers | (observers or set())

    def __get_action_request_handler[State, Context](
        self,
        action_request_handler: PipelineActionRequestHandler[State, Context] | None = None,
    ) -> PipelineActionRequestHandler[State, Context] | None:
        """Get the action request handler to use for the execution."""
        return action_request_handler or self._default_action_request_handler
