from typing import Generic

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.state import State
from experiments.lib.pipelines.tasks import Task, TaskResult


class Step(Generic[State, Context]):
    """Step represents a single step in a pipeline.

    Each step has a name and a task that defines the operation to be performed. The task is called with the current state and a context, and returns the updated state.

    Args:
        name: The name of the step.
        task: The task to execute in this step.
    """

    def __init__(self, name, task: Task[State, Context]) -> None:
        self.name = name
        self.task = task

    def __repr__(self) -> str:
        return f"Step(name={self.name})"

    def __call__(self, state: State, context: Context) -> TaskResult[State]:
        return self.task(state, context)
