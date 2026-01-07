"""Defines the Task protocol and related classes for pipeline tasks.

A Task represents a unit of work in a pipeline that processes a given state and returns an updated state.
"""

import enum
from typing import Generic, NamedTuple, Protocol

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.state import State


class TaskStatus(enum.Enum):
    """Enumeration representing the possible outcomes of a pipeline task.

    - `SUCCESS`: The task completed successfully.
    - `FAILURE`: The task failed to complete (due to missing data, for example).
    - `SKIPPED`: The task was skipped.
    - `ERROR`: An error occurred during the execution of the task.
    - `REQUIRES_ACTION`: The task requires user action to proceed.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"


class TaskResult(Generic[State], NamedTuple):
    """Result of executing a task in a pipeline.

    It contains the updated state after the step execution.

    Args:
        state: The updated state after executing the step.
        status: The status of the step execution.
        message: An optional message providing additional information about the step execution.
        error: An optional `PipelineException` if an error occurred during step execution.
    """

    state: State
    status: TaskStatus
    message: str | None = None
    error: Exception | None = None


class Task(Generic[State, Context], Protocol):
    """Task is a protocol that defines the contract for anything that can be executed in a pipeline.

    It takes the current state of type `State` and returns a new state of the same type `State`.
    """

    def __call__(self, state: State, context: Context) -> TaskResult[State]:
        """Invoke the task with the given state and context.

        Args:
            state: The current state to process.
            context: Context for the task.

        Returns:
            The updated state after processing.

        Raises:
            PipelineException: If there is an error during execution.
        """
        ...
