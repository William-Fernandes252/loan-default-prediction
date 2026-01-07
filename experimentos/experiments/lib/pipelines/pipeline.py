from typing import Generic

from experiments.lib.pipelines.context import Context
from experiments.lib.pipelines.state import State
from experiments.lib.pipelines.steps import Step, Task


class Pipeline(Generic[State, Context]):
    """Pipeline represents a sequence of steps to be executed in order.

    A pipeline consists of multiple steps, each represented by a `Step` object. The `run` method executes each step in sequence, passing the state from one step to the next.

    Args:
        name: The name of the pipeline.
        context: The context for the pipeline.
    """

    _steps: list[Step[State, Context]]

    def __init__(self, name: str, context: Context) -> None:
        self._name = name
        self._context = context
        self._steps = []

    def add_step(self, name: str, task: Task[State, Context]) -> None:
        """Add a step to the pipeline by name and runnable.
        Args:
            name: The name of the step.
            task: The task operation for the step.
        """
        step = Step(name, task)
        self._steps.append(step)

    def __repr__(self):
        step_names = ", ".join(step.name for step in self.steps)
        return f"{self.name}(steps=[{step_names}])"

    @property
    def steps(self) -> list[Step[State, Context]]:
        return self._steps

    @property
    def context(self) -> Context:
        return self._context

    @property
    def name(self) -> str:
        return self._name
