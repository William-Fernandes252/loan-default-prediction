from typing import Annotated, Callable, Literal, NamedTuple, TypedDict

from experiments.lib.pipelines.steps import Step, Task

type StepCondition[State, Context] = Callable[
    [State, Context], tuple[Literal[True], None] | tuple[Literal[False], str]
]
"""Condition to determine if a step should execute.

It takes the current state and context as input and returns a tuple where the first element is a boolean indicating whether to execute the step, and the second element is an optional reason for skipping the step.
"""


class StepConfig(TypedDict, total=False):
    """Configuration for a pipeline step."""

    max_retries: Annotated[int, "Maximum number of retries for the step"]
    auto_retry: Annotated[bool, "Whether to automatically retry on failure"]


class PipelineStep[State, Context](NamedTuple):
    """Represents a step in the pipeline along with its configuration."""

    step: Step[State, Context]
    config: StepConfig
    condition: StepCondition[State, Context] | None = None


class Pipeline[State, Context]:
    """Pipeline represents a sequence of steps to be executed in order.

    A pipeline consists of multiple steps, each represented by a `Step` object. The `run` method executes each step in sequence, passing the state from one step to the next.

    Args:
        name: The name of the pipeline.
        context: The context for the pipeline.

    """

    _steps: list[PipelineStep[State, Context]]

    def __init__(self, name: str, context: Context) -> None:
        self._name = name
        self._context = context
        self._steps = []

    def add_step(
        self, name: str, task: Task[State, Context], config: StepConfig | None = None
    ) -> None:
        """Add a step to the pipeline by name and runnable.

        Args:
            name: The name of the step.
            task: The task operation for the step.
            config: Optional configuration for the step.
        """
        step = Step(name, task)
        self._add_pipeline_step(step, config)

    def add_conditional_step(
        self,
        name: str,
        task: Task[State, Context],
        condition: StepCondition[State, Context],
        config: StepConfig | None = None,
    ) -> None:
        """Add a conditional step to the pipeline.

        Args:
            name: The name of the step.
            task: The task operation for the step.
            condition: The condition to determine if the step should execute.
            config: Optional configuration for the step.
        """
        step = Step(name, task)
        self._add_pipeline_step(step, config=config, condition=condition)

    def _add_pipeline_step(
        self,
        step: Step[State, Context],
        config: StepConfig | None = None,
        condition: StepCondition[State, Context] | None = None,
    ) -> None:
        config = config or {}
        config.update(_default_step_config)

        self._steps.append(PipelineStep(step, config, condition))

    def __repr__(self):
        step_names = ", ".join(ps.step.name for ps in self.steps)
        return f"{self.name}(steps=[{step_names}])"

    @property
    def steps(self) -> list[PipelineStep[State, Context]]:
        """Get the steps of the pipeline."""
        return self._steps

    @property
    def context(self) -> Context:
        return self._context

    @property
    def name(self) -> str:
        return self._name


_default_step_config: StepConfig = {
    "max_retries": 3,
    "auto_retry": False,
}


def set_step_config_defaults(config: StepConfig) -> None:
    """Set the default configuration for pipeline steps.

    Args:
        config: The default step configuration to set.
    """
    _default_step_config.update(config)
