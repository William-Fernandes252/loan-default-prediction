from experiments.core.modeling.schema import ExperimentConfig


def test_experiment_config_defaults() -> None:
    config = ExperimentConfig(cv_folds=3, cost_grids=[{"c": 1}])

    assert config.cv_folds == 3
    assert config.cost_grids == [{"c": 1}]
    assert config.discard_checkpoints is False
