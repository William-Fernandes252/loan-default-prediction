from dataclasses import dataclass
from enum import Enum
import time
from typing import Annotated, Generic, Protocol

from loguru import logger

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.errors import PipelineException, PipelineInterruption
from experiments.lib.pipelines.pipeline import Pipeline
from experiments.lib.pipelines.state import State
from experiments.lib.pipelines.steps import Step


class ErrorAction(Enum):
    """Actions to take when an error occurs during pipeline execution.

    `ABORT`: Stop the pipeline execution immediately.
    `RETRY`: Retry the failed step.
    `SKIP`: Skip the failed step and continue with the next one.
    `IGNORE`: Ignore the error and continue execution as if nothing happened.
    """

    ABORT = "abort"
    RETRY = "retry"
    SKIP = "skip"
    IGNORE = "ignore"


class ErrorHandler(Protocol):
    def __call__(
        self, step_name: str, exception: PipelineException | PipelineInterruption
    ) -> ErrorAction:
        """Handle an error that occurred during pipeline execution.

        It can be used to log the error, perform cleanup, prompt the user, or take other appropriate actions. The handler should return True if the pipeline execution should continue, or False to abort.

        Args:
            step_name: The name of the step where the error occurred.
            exception: The exception that was raised.
        """
        ...


@dataclass
class PipelineExecutionResult(Generic[State, Context]):
    """Result of pipeline execution."""

    duration: Annotated[dict[str, float], "Duration of each step in seconds."]
    context: Annotated[Context, "The context used during pipeline execution."]
    final_state: Annotated[State, "Final state after pipeline execution."]
    last_executed_step: Annotated[str | None, "Name of the last executed step."]
    errors: Annotated[
        dict[str, PipelineException],
        "Mapping of step names to error messages for failed steps.",
    ]

    def total_duration(self) -> float:
        """Calculate the total duration of the pipeline execution."""
        return sum(self.duration.values())

    def succeeded(self) -> bool:
        """Check if the pipeline execution succeeded without errors."""
        return len(self.errors) == 0

    def last_error(self) -> PipelineException | None:
        """Get the last error that occurred during pipeline execution, if any."""
        if not self.errors:
            return None
        last_step = self.last_executed_step
        if last_step and last_step in self.errors:
            return self.errors[last_step]
        return None


class PipelineExecutor:
    """Orchestrates the execution of pipeline steps."""

    @staticmethod
    def _get_step_id(step: Step, pipeline: Pipeline) -> str:
        """Get a unique identifier for a step within a pipeline.

        It is used for logging and error reporting.
        """
        return f"{pipeline.name}.{step.name}"

    def execute[State, Context](
        self,
        pipeline: Pipeline[State, Context],
        initial_state: State,
        error_handlers: dict[str, ErrorHandler] | None = None,
    ) -> PipelineExecutionResult[State, Context]:
        """Execute the pipeline steps sequentially.

        Args:
            pipeline: The pipeline to execute.
            initial_state: The initial state to start the pipeline with.
            error_handlers: Optional mapping of step names to error handler functions.

        Returns:
            Result object containing execution details.

        Raises:
            errors.PipelineException: If a step fails and no error handler is provided.
        """

        logger.info(f"Starting pipeline execution for: {pipeline.name}")
        start_time = time.time()

        last_executed_step: str | None = None
        step_durations: dict[str, float] = {}
        errors_dict: dict[str, PipelineException] = {}
        context = pipeline.context
        steps = pipeline.steps
        current_state = initial_state
        for step in steps:
            step_id = self._get_step_id(step, pipeline)
            step_start_time = time.time()
            try:
                logger.info(f"Starting step: {step_id}")
                current_state, *_ = step(current_state, context)
            except (PipelineException, PipelineInterruption) as e:
                if not isinstance(e, PipelineInterruption):
                    errors_dict[step.name] = e

                if error_handlers and step.name in error_handlers:
                    should_continue = error_handlers[step.name](step.name, e)
                    if not should_continue:
                        logger.info(
                            f"Pipeline execution aborted by error handler at step '{step_id}'."
                        )
                        break
                else:
                    logger.error(f"Step '{step_id}' failed with error: {e}. Aborting pipeline.")
                    break
            else:
                logger.success(f"Step '{step_id}' succeeded.")
            finally:
                end_time = time.time()
                duration = end_time - step_start_time
                step_durations[step.name] = duration
                logger.info(f"Step '{step_id}' completed in {duration:.2f}s")
                last_executed_step = step.name

        duration = time.time() - start_time
        logger.info(f"Pipeline finished in {duration:.2f}s")

        return PipelineExecutionResult(
            duration=step_durations,
            final_state=current_state,
            last_executed_step=last_executed_step,
            errors=errors_dict,
            context=pipeline.context,
        )
