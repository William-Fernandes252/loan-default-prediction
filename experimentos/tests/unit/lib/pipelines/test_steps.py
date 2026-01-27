from experiments.lib.pipelines.steps import Step
from experiments.lib.pipelines.tasks import TaskResult, TaskStatus


class DescribeStep:
    def it_runs_task_on_call(self):
        def sample_task(state: int, context: dict) -> TaskResult[int]:
            new_state = state + context["increment"]
            return TaskResult(new_state, TaskStatus.SUCCESS)

        step = Step(name="sample_step", task=sample_task)

        initial_state = 5
        context = {"increment": 3}
        result = step(initial_state, context)

        assert result.state == 8
