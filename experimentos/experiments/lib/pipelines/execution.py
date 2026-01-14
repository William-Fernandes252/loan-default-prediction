"""Defines the pipeline execution engine with managed parallel execution."""

from collections.abc import Sequence, Set
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Annotated, Any, final

from experiments.lib.pipelines.lifecycle import (
    Action,
    PipelineObserver,
)
from experiments.lib.pipelines.pipeline import Pipeline, PipelineStep, StepConfig
from experiments.lib.pipelines.steps import Step
from experiments.lib.pipelines.tasks import TaskResult, TaskStatus

type _AnyPipeline = Pipeline[Any, Any]
"""Type alias for a pipeline with any state and context types."""

type _AnyTaskResult = TaskResult[Any]
"""Type alias for a task result with any state type."""


class PipelineStatus(Enum):
    """Status of a pipeline in the executor."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    PANICKED = "panicked"


@dataclass
class StepTrace:
    """Trace information for a single pipeline step execution."""

    name: Annotated[str, "The name of the step."]
    duration: Annotated[float, "Duration of the step execution in seconds."]
    status: Annotated[TaskStatus, "The status of the step execution."]
    message: Annotated[str | None, "The message returned from the step execution, if any."]
    error: Annotated[Exception | None, "The exception raised during step execution, if any."]


@dataclass
class PipelineExecutionResult[State, Context]:
    """Result of pipeline execution."""

    pipeline_name: Annotated[str, "Name of the pipeline."]
    status: Annotated[PipelineStatus, "Final status of the pipeline."]
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
        return self.status == PipelineStatus.COMPLETED and len(self.errors) == 0

    def last_error(self) -> Exception | None:
        """Get the last error that occurred during pipeline execution, if any."""
        if not self.errors:
            return None
        last_step = self.last_executed_step
        if last_step and last_step in self.errors:
            return self.errors[last_step]
        return None


@dataclass
class _PipelineContext:
    """Internal context for tracking pipeline execution state."""

    pipeline: _AnyPipeline
    initial_state: Any
    current_state: Any
    context: Any
    step_index: int = 0
    step_traces: list[StepTrace] = field(default_factory=list)
    step_retry_counts: dict[int, int] = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.PENDING
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def current_pipeline_step(self) -> PipelineStep[Any, Any] | None:
        """Get the current pipeline step tuple (step, config), or None if finished."""
        if self.step_index >= len(self.pipeline.steps):
            return None
        return self.pipeline.steps[self.step_index]

    @property
    def current_step(self) -> Step[Any, Any] | None:
        """Get the current step to execute, or None if finished."""
        ps = self.current_pipeline_step
        return ps.step if ps else None

    @property
    def current_step_config(self) -> StepConfig | None:
        """Get the configuration for the current step."""
        ps = self.current_pipeline_step
        return ps.config if ps else None

    def advance(self) -> None:
        """Advance to the next step."""
        self.step_index += 1

    def reset(self) -> None:
        """Reset the pipeline to initial state for retry."""
        self.step_index = 0
        self.current_state = self.initial_state
        self.step_traces = []
        self.step_retry_counts = {}
        self.status = PipelineStatus.RUNNING

    def is_finished(self) -> bool:
        """Check if all steps have been executed."""
        return self.step_index >= len(self.pipeline.steps)

    def get_retry_count(self) -> int:
        """Get the current retry count for the current step."""
        return self.step_retry_counts.get(self.step_index, 0)

    def increment_retry_count(self) -> int:
        """Increment and return the retry count for the current step."""
        count = self.step_retry_counts.get(self.step_index, 0) + 1
        self.step_retry_counts[self.step_index] = count
        return count

    def can_retry(self) -> bool:
        """Check if the current step can be retried based on StepConfig."""
        config = self.current_step_config
        if not config:
            return False
        max_retries = config.get("max_retries", 3)
        return self.get_retry_count() < max_retries

    def should_auto_retry(self) -> bool:
        """Check if the current step should automatically retry on error."""
        config = self.current_step_config
        if not config:
            return False
        return config.get("auto_retry", False) and self.can_retry()


@dataclass
class _StepExecution:
    """Represents a step execution unit for the thread pool."""

    pipeline_ctx: _PipelineContext
    step: Step[Any, Any]
    state: Any
    context: Any


@dataclass
class _StepResult:
    """Result of a step execution from the thread pool."""

    pipeline_ctx: _PipelineContext
    step_name: str
    task_result: _AnyTaskResult | None
    exception: Exception | None
    duration: float


@final
class PipelineExecutor:
    """Orchestrates parallel execution of multiple pipelines.

    The executor manages a pool of worker threads that execute pipeline steps
    concurrently. Steps from different pipelines can run in parallel, while
    steps within a single pipeline are executed sequentially.

    All control flow (retries, aborts, panics) is managed through lifecycle
    hooks provided by observers. Observers can be provided at construction
    time (default observers) and/or at execution time.

    Args:
        max_workers: Maximum number of worker threads. Defaults to 4.
        observers: A set of default observers to notify during execution.
        default_action: The default action to take if no observer provides
            guidance. Default is `Action.ABORT`.
    """

    def __init__(
        self,
        max_workers: int = 4,
        observers: Set[PipelineObserver[Any, Any]] | None = None,
        default_action: Action = Action.ABORT,
    ) -> None:
        self._max_workers = max_workers
        self._default_observers: Set[PipelineObserver[Any, Any]] = observers or set()
        self._default_action = default_action

        self._executor: ThreadPoolExecutor | None = None
        self._pipeline_contexts: dict[str, _PipelineContext] = {}
        self._pending_futures: dict[Future[_StepResult], str] = {}
        self._results: list[PipelineExecutionResult[Any, Any]] = []
        self._panic_error: Exception | None = None

        # Observers for current execution (merged with defaults)
        self._active_observers: Set[PipelineObserver[Any, Any]] = set()

        self._lock = threading.Lock()
        self._completion_event = threading.Event()
        self._started = False
        self._shutdown = False

    def schedule(
        self,
        pipeline: _AnyPipeline,
        initial_state: Any,
        context: Any,
    ) -> None:
        """Schedule a pipeline for execution.

        The pipeline will be queued and executed when `start()` is called.
        Multiple pipelines can be scheduled before starting execution.

        Args:
            pipeline: The pipeline to execute.
            initial_state: The initial state to start the pipeline with.
            context: The context for the pipeline.

        Raises:
            RuntimeError: If the executor has already been started.
        """
        if self._started:
            raise RuntimeError("Cannot schedule pipelines after execution has started.")

        pipeline_id = f"{pipeline.name}_{id(pipeline)}"
        ctx = _PipelineContext(
            pipeline=pipeline,
            initial_state=initial_state,
            current_state=initial_state,
            context=context,
        )
        self._pipeline_contexts[pipeline_id] = ctx

    def start(
        self,
        observers: Set[PipelineObserver[Any, Any]] | None = None,
    ) -> None:
        """Start execution of all scheduled pipelines.

        This method is non-blocking. Use `wait()` to block until all
        pipelines have completed.

        Args:
            observers: Optional set of observers to use in addition to
                the default observers.

        Raises:
            RuntimeError: If the executor has already been started.
        """
        if self._started:
            raise RuntimeError("Executor has already been started.")

        self._started = True
        self._active_observers = self._default_observers | (observers or set())
        self._completion_event.clear()

        if not self._pipeline_contexts:
            self._completion_event.set()
            return

        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # Submit initial steps for all pipelines
        for pipeline_id, ctx in self._pipeline_contexts.items():
            self._start_pipeline(pipeline_id, ctx)

    def wait(self) -> Sequence[PipelineExecutionResult[Any, Any]]:
        """Wait for all pipelines to complete and return results.

        This method blocks until all scheduled pipelines have finished
        execution. It does not raise exceptions for pipeline failures
        unless a PANIC action was requested by an observer.

        Returns:
            A sequence of execution results for all scheduled pipelines.

        Raises:
            Exception: If any observer requested a PANIC action, the
                original exception is re-raised.
        """
        self._completion_event.wait()

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._panic_error:
            raise self._panic_error

        return self._results

    def execute[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        initial_state: State,
        context: Context,
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> PipelineExecutionResult[State, Context]:
        """Execute a single pipeline synchronously.

        This is a convenience method that schedules, starts, and waits
        for a single pipeline.

        Args:
            pipeline: The pipeline to execute.
            initial_state: The initial state to start the pipeline with.
            context: The context for the pipeline.
            observers: Optional set of observers to use in addition to
                the default observers.

        Returns:
            The execution result for the pipeline.
        """
        self.schedule(pipeline, initial_state, context)
        self.start(observers)
        results = self.wait()
        return results[0]  # type: ignore[return-value]

    def execute_all[State, Context](
        self,
        pipelines: Sequence[tuple[Pipeline[State, Context], State]],
        context: Context,
        observers: Set[PipelineObserver[State, Context]] | None = None,
    ) -> list[PipelineExecutionResult[State, Context]]:
        """Execute multiple pipelines in parallel.

        This is a convenience method that schedules all pipelines, starts
        execution, and waits for completion.

        Args:
            pipelines: A sequence of (pipeline, initial_state) tuples.
            observers: Optional set of observers to use in addition to
                the default observers.

        Returns:
            A sequence of execution results for all pipelines.
        """
        for pipeline, state in pipelines:
            self.schedule(pipeline, state, context)
        self.start(observers)
        return self.wait()  # type: ignore[return-value]

    def _start_pipeline(
        self,
        pipeline_id: str,
        ctx: _PipelineContext,
    ) -> None:
        """Start execution of a pipeline."""
        ctx.status = PipelineStatus.RUNNING

        action = self._notify_pipeline_start(ctx.pipeline)

        if action == Action.PANIC:
            error = RuntimeError(f"Pipeline {ctx.pipeline.name} start was rejected with PANIC")
            self._handle_panic(error, ctx)
            return

        if action == Action.ABORT:
            self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
            return

        self._submit_next_step(pipeline_id, ctx)

    def _submit_next_step(
        self,
        pipeline_id: str,
        ctx: _PipelineContext,
    ) -> None:
        """Submit the next step of a pipeline for execution."""
        if self._shutdown or self._panic_error:
            self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
            return

        pipeline_step = ctx.current_pipeline_step
        if pipeline_step is None:
            self._complete_pipeline(pipeline_id, ctx)
            return

        step = pipeline_step.step
        condition = pipeline_step.condition

        # Evaluate condition if present
        if condition is not None:
            try:
                should_execute, skip_reason = condition(ctx.current_state, ctx.context)
            except Exception as e:
                self._handle_step_error(pipeline_id, ctx, step.name, e)
                return

            if not should_execute:
                # Step should be skipped
                action = self._notify_step_skipped(
                    ctx.pipeline, step.name, skip_reason or "Condition not met"
                )

                if action == Action.PANIC:
                    error = RuntimeError(f"Step {step.name} skip triggered PANIC")
                    self._handle_panic(error, ctx)
                    return

                if action == Action.ABORT:
                    self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
                    return

                # PROCEED - advance to next step
                with ctx.lock:
                    ctx.advance()
                self._submit_next_step(pipeline_id, ctx)
                return

        # Notify step start and check for abort/panic
        action = self._notify_step_start(ctx.pipeline, step.name, ctx.current_state)

        if action == Action.PANIC:
            error = RuntimeError(f"Step {step.name} start was rejected with PANIC")
            self._handle_panic(error, ctx)
            return

        if action == Action.ABORT:
            self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
            return

        execution = _StepExecution(
            pipeline_ctx=ctx,
            step=step,
            state=ctx.current_state,
            context=ctx.context,
        )

        future = self._executor.submit(self._execute_step, execution)  # type: ignore[union-attr]
        future.add_done_callback(lambda f: self._on_step_complete(pipeline_id, f))

        with self._lock:
            self._pending_futures[future] = pipeline_id

    def _execute_step(
        self,
        execution: _StepExecution,
    ) -> _StepResult:
        """Execute a single step in a worker thread."""
        start_time = time.time()
        task_result: _AnyTaskResult | None = None
        exception: Exception | None = None

        try:
            task_result = execution.step(execution.state, execution.context)
        except Exception as e:
            exception = e

        duration = time.time() - start_time

        return _StepResult(
            pipeline_ctx=execution.pipeline_ctx,
            step_name=execution.step.name,
            task_result=task_result,
            exception=exception,
            duration=duration,
        )

    def _on_step_complete(
        self,
        pipeline_id: str,
        future: Future[_StepResult],
    ) -> None:
        """Handle completion of a step execution."""
        with self._lock:
            self._pending_futures.pop(future, None)

        try:
            result = future.result()
        except Exception as e:
            # Future itself failed (shouldn't happen normally)
            ctx = self._pipeline_contexts.get(pipeline_id)
            if ctx:
                self._handle_panic(e, ctx)
            return

        ctx = result.pipeline_ctx
        step_name = result.step_name

        # Record trace
        trace = StepTrace(
            name=step_name,
            duration=result.duration,
            status=(result.task_result.status if result.task_result else TaskStatus.ERROR),
            message=result.task_result.message if result.task_result else None,
            error=result.exception or (result.task_result.error if result.task_result else None),
        )

        with ctx.lock:
            ctx.step_traces.append(trace)

        # Handle exception from step execution
        if result.exception:
            self._handle_step_error(pipeline_id, ctx, step_name, result.exception)
            return

        # Handle task result
        task_result = result.task_result
        assert task_result is not None

        if task_result.status == TaskStatus.ERROR and task_result.error:
            self._handle_step_error(pipeline_id, ctx, step_name, task_result.error)
            return

        # Success path - notify observers and check their response
        action = self._notify_step_finish(ctx.pipeline, step_name, task_result)

        if action == Action.PANIC:
            error = RuntimeError(f"Step {step_name} finish triggered PANIC")
            self._handle_panic(error, ctx)
            return

        if action == Action.ABORT:
            self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
            return

        if action == Action.RETRY:
            # Re-execute this step (don't advance)
            with ctx.lock:
                if not ctx.can_retry():
                    # Max retries exceeded, abort
                    self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
                    return
                ctx.increment_retry_count()
            self._submit_next_step(pipeline_id, ctx)
            return

        # PROCEED - advance to next step
        with ctx.lock:
            ctx.current_state = task_result.state
            ctx.advance()

        self._submit_next_step(pipeline_id, ctx)

    def _handle_step_error(
        self,
        pipeline_id: str,
        ctx: _PipelineContext,
        step_name: str,
        error: Exception,
    ) -> None:
        """Handle an error that occurred during step execution."""
        # Check auto_retry before consulting observers (only for errors)
        with ctx.lock:
            if ctx.should_auto_retry():
                ctx.increment_retry_count()
                self._submit_next_step(pipeline_id, ctx)
                return

        action = self._notify_error(ctx.pipeline, step_name, error)

        if action == Action.PANIC:
            self._handle_panic(error, ctx)
            return

        if action == Action.ABORT:
            self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
            return

        if action == Action.PROCEED:
            # Skip the failed step and continue
            with ctx.lock:
                ctx.advance()
            self._submit_next_step(pipeline_id, ctx)
            return

        if action == Action.RETRY:
            with ctx.lock:
                if not ctx.can_retry():
                    # Max retries exceeded, abort
                    self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
                    return
                ctx.increment_retry_count()
            self._submit_next_step(pipeline_id, ctx)
            return

    def _complete_pipeline(
        self,
        pipeline_id: str,
        ctx: _PipelineContext,
    ) -> None:
        """Complete a pipeline and check observer response for retry."""
        result = PipelineExecutionResult(
            pipeline_name=ctx.pipeline.name,
            status=PipelineStatus.COMPLETED,
            step_traces=ctx.step_traces,
            context=ctx.context,
            final_state=ctx.current_state,
        )

        action = self._notify_pipeline_finish(ctx.pipeline, result)

        if action == Action.PANIC:
            error = RuntimeError(f"Pipeline {ctx.pipeline.name} finish triggered PANIC")
            self._handle_panic(error, ctx)
            return

        if action == Action.RETRY:
            # Re-execute the entire pipeline from the beginning
            with ctx.lock:
                ctx.reset()
            self._notify_pipeline_start(ctx.pipeline)
            self._submit_next_step(pipeline_id, ctx)
            return

        if action == Action.ABORT:
            self._finalize_pipeline(ctx, PipelineStatus.ABORTED)
            return

        # PROCEED - finalize normally
        self._record_result(result)

    def _handle_panic(
        self,
        error: Exception,
        ctx: _PipelineContext,
    ) -> None:
        """Handle a PANIC action by shutting down all execution."""
        with self._lock:
            if self._panic_error is not None:
                return  # Already panicking

            self._panic_error = error
            self._shutdown = True
            ctx.status = PipelineStatus.PANICKED

        # Finalize remaining pipelines
        self._finalize_all_remaining()

    def _finalize_pipeline(
        self,
        ctx: _PipelineContext,
        status: PipelineStatus,
    ) -> None:
        """Finalize a pipeline and record its result."""
        with ctx.lock:
            if ctx.status not in (PipelineStatus.PENDING, PipelineStatus.RUNNING):
                return  # Already finalized
            ctx.status = status

        result = PipelineExecutionResult(
            pipeline_name=ctx.pipeline.name,
            status=status,
            step_traces=ctx.step_traces,
            context=ctx.context,
            final_state=ctx.current_state,
        )

        # Notify but don't act on response for aborted/panicked pipelines
        self._notify_pipeline_finish(ctx.pipeline, result)
        self._record_result(result)

    def _record_result(
        self,
        result: PipelineExecutionResult[Any, Any],
    ) -> None:
        """Record a pipeline result and check for completion."""
        with self._lock:
            self._results.append(result)
            self._check_completion()

    def _finalize_all_remaining(self) -> None:
        """Finalize all pipelines that haven't completed yet."""
        for pipeline_id, ctx in self._pipeline_contexts.items():
            with ctx.lock:
                if ctx.status in (PipelineStatus.PENDING, PipelineStatus.RUNNING):
                    self._finalize_pipeline(ctx, PipelineStatus.ABORTED)

    def _check_completion(self) -> None:
        """Check if all pipelines have completed and signal if so."""
        if len(self._results) >= len(self._pipeline_contexts):
            self._completion_event.set()

    # Observer notification methods

    def _notify_pipeline_start(
        self,
        pipeline: _AnyPipeline,
    ) -> Action:
        """Notify observers that a pipeline is starting."""
        if not self._active_observers:
            return Action.PROCEED

        actions: list[Action] = []
        for observer in self._active_observers:
            try:
                action = observer.on_pipeline_start(pipeline)
                actions.append(action)
            except Exception:
                pass  # Don't let observer errors affect execution

        if not actions:
            return Action.PROCEED

        return max(actions, key=lambda a: a.value)

    def _notify_pipeline_finish(
        self,
        pipeline: _AnyPipeline,
        result: PipelineExecutionResult[Any, Any],
    ) -> Action:
        """Notify observers that a pipeline has finished."""
        if not self._active_observers:
            return Action.PROCEED

        actions: list[Action] = []
        for observer in self._active_observers:
            try:
                action = observer.on_pipeline_finish(pipeline, result)
                actions.append(action)
            except Exception:
                pass

        if not actions:
            return Action.PROCEED

        return max(actions, key=lambda a: a.value)

    def _notify_step_start(
        self,
        pipeline: _AnyPipeline,
        step_name: str,
        current_state: Any,
    ) -> Action:
        """Notify observers that a step is starting."""
        if not self._active_observers:
            return Action.PROCEED

        actions: list[Action] = []
        for observer in self._active_observers:
            try:
                action = observer.on_step_start(pipeline, step_name, current_state)
                actions.append(action)
            except Exception:
                pass

        if not actions:
            return Action.PROCEED

        return max(actions, key=lambda a: a.value)

    def _notify_step_finish(
        self,
        pipeline: _AnyPipeline,
        step_name: str,
        step_result: _AnyTaskResult,
    ) -> Action:
        """Notify observers that a step has finished."""
        if not self._active_observers:
            return Action.PROCEED

        actions: list[Action] = []
        for observer in self._active_observers:
            try:
                action = observer.on_step_finish(pipeline, step_name, step_result)
                actions.append(action)
            except Exception:
                pass

        if not actions:
            return Action.PROCEED

        return max(actions, key=lambda a: a.value)

    def _notify_step_skipped(
        self,
        pipeline: _AnyPipeline,
        step_name: str,
        reason: str,
    ) -> Action:
        """Notify observers that a step was skipped due to a condition."""
        if not self._active_observers:
            return Action.PROCEED

        actions: list[Action] = []
        for observer in self._active_observers:
            try:
                action = observer.on_step_skipped(pipeline, step_name, reason)
                actions.append(action)
            except Exception:
                pass

        if not actions:
            return Action.PROCEED

        return max(actions, key=lambda a: a.value)

    def _notify_error(
        self,
        pipeline: _AnyPipeline,
        step_name: str,
        error: Exception,
    ) -> Action:
        """Notify observers of an error and collect their response.

        Returns the highest-priority action among all observers.
        """
        if not self._active_observers:
            return self._default_action

        actions: list[Action] = []
        for observer in self._active_observers:
            try:
                action = observer.on_error(pipeline, step_name, error)
                actions.append(action)
            except Exception:
                pass

        if not actions:
            return self._default_action

        return max(actions, key=lambda a: a.value)
