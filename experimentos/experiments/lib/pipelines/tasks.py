"""Defines the Task protocol and related classes for pipeline tasks.

A Task represents a unit of work in a pipeline. It is a callable that takes the current `State` and `Context`, performs some operation, and returns an updated state along with a status indicating the outcome of the operation.

The `State` can be any type, allowing for flexibility in defining different pipeline states. It is mutable, meaning that pipeline steps can modify and return updated versions of the state as needed.

The `Context` is also a generic type variable that can be any type, allowing for flexibility in defining different pipeline contexts. It is immutable (read-only), meaning that pipeline steps should not modify the context directly, and it should be used only for things like configuration or shared resources.
"""

import enum
from typing import NamedTuple, Protocol


class TaskStatus(enum.Enum):
    """Enumeration representing the possible outcomes of a pipeline task.

    - `SUCCESS`: The task completed successfully.
    - `FAILURE`: The task failed to complete (due to missing data, for example).
    - `ERROR`: An error occurred during the execution of the task.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class TaskResult[State](NamedTuple):
    """Result of executing a task in a pipeline.

    It contains the updated state after the step execution.

    Args:
        state: The updated state after executing the step.
        status: The status of the step execution.
        message: An optional message providing additional information about the step execution.
        error: An optional `Exception` if an error occurred during step execution.
    """

    state: State
    status: TaskStatus
    message: str | None = None
    error: Exception | None = None


class Task[State, Context](Protocol):
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
            Exception: If there is an error during execution.
        """
        ...
