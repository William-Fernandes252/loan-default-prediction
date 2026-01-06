from typing import Generic, Protocol

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.state import State


class Runnable(Generic[State, Context], Protocol):
    """Runnable is a protocol that defines the contract for anything that can be executed in a pipeline.

    It takes the current state of type `State` and returns a new state of the same type `State`.
    """

    def __call__(self, state: State, context: Context) -> State:
        """Invoke the runnable with the given state.

        Args:
            state: The current state to process.
            context: Context for the runnable.

        Returns:
            The updated state after processing.

        Raises:
            PipelineException: If there is an error during execution.
        """
        ...


class Step(Generic[State, Context]):
    """Step represents a single step in a pipeline.

    Each step has a name and a runnable that defines the operation to be performed. The runnable is called with the current state and a context, and returns the updated state.

    Args:
        name: The name of the step.
        runnable: The runnable operation to execute in this step.
    """

    def __init__(self, name, runnable: Runnable[State, Context]) -> None:
        self.name = name
        self.runnable = runnable

    def __repr__(self) -> str:
        return f"Step(name={self.name})"

    def __call__(self, state: State, context: Context) -> State:
        return self.runnable(state, context)
