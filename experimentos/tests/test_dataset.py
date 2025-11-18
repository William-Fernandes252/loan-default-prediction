from unittest.mock import Mock

import pytest
import typer

from experiments import dataset as dataset_module
from experiments.config import Dataset
from experiments.utils import overwrites as overwrites_module


@pytest.fixture
def parallel_spy(monkeypatch):
    """Mocks joblib.Parallel/delayed while recording submissions."""

    class ParallelStub:
        def __init__(self):
            self.args = None
            self.kwargs = None
            self.submitted = []
            self.call_count = 0

        def __call__(self, tasks):
            self.call_count += 1
            submissions = list(tasks)
            self.submitted.extend(submissions)
            return [task() for task in submissions]

    stub = ParallelStub()

    def fake_parallel(*args, **kwargs):
        stub.args = args
        stub.kwargs = kwargs
        stub.submitted = []
        stub.call_count = 0
        return stub

    parallel_ctor = Mock(side_effect=fake_parallel)
    monkeypatch.setattr(dataset_module, "Parallel", parallel_ctor)

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(dataset_module, "delayed", fake_delayed)
    stub.constructor = parallel_ctor
    return stub


def test_process_single_dataset_success(monkeypatch, tmp_path):
    dataset = Dataset.LENDING_CLUB
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.parquet"
    extra_params = {"example": 1}

    raw_df = object()
    processed_df = Mock()
    processed_df.write_parquet = Mock()

    read_csv_mock = Mock(return_value=raw_df)
    monkeypatch.setattr(dataset_module.pl, "read_csv", read_csv_mock)

    monkeypatch.setattr(
        Dataset,
        "get_path",
        lambda self: raw_path,
        raising=False,
    )
    monkeypatch.setattr(
        dataset_module,
        "get_processed_path",
        lambda ds: processed_path,
    )
    monkeypatch.setattr(
        Dataset,
        "get_extra_params",
        lambda self: extra_params,
        raising=False,
    )

    def fake_process(self, raw_data):
        assert raw_data is raw_df
        return processed_df

    monkeypatch.setattr(Dataset, "process_data", fake_process, raising=False)

    result = dataset_module._process_single_dataset(dataset)

    assert result == (dataset, True, None)
    read_csv_mock.assert_called_once()
    args, kwargs = read_csv_mock.call_args
    assert args == (raw_path,)
    expected_kwargs = {"low_memory": False, "use_pyarrow": True, **extra_params}
    assert kwargs == expected_kwargs
    processed_df.write_parquet.assert_called_once_with(processed_path)


def test_process_single_dataset_failure_returns_error_message(monkeypatch, tmp_path):
    dataset = Dataset.CORPORATE_CREDIT_RATING
    raw_path = tmp_path / "corp.csv"
    processed_path = tmp_path / "corp.parquet"

    monkeypatch.setattr(Dataset, "get_path", lambda self: raw_path, raising=False)
    monkeypatch.setattr(dataset_module, "get_processed_path", lambda ds: processed_path)
    monkeypatch.setattr(Dataset, "get_extra_params", lambda self: {}, raising=False)
    monkeypatch.setattr(dataset_module.pl, "read_csv", Mock(return_value=object()))

    def failing_process(self, raw_data):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(Dataset, "process_data", failing_process, raising=False)

    dataset_value, success, error = dataset_module._process_single_dataset(dataset)

    assert dataset_value is dataset
    assert success is False
    assert "boom" in error


def _mock_processed_path(monkeypatch, tmp_path):
    def fake_path(dataset):
        return tmp_path / f"{dataset.value}.parquet"

    monkeypatch.setattr(dataset_module, "get_processed_path", fake_path)


def test_main_processes_specific_dataset_in_parallel(parallel_spy, monkeypatch, tmp_path):
    calls: list[Dataset] = []

    def fake_process(dataset: Dataset):
        calls.append(dataset)
        return (dataset, True, None)

    monkeypatch.setattr(dataset_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(dataset_module, "cpu_count", lambda: 4)
    _mock_processed_path(monkeypatch, tmp_path)
    monkeypatch.setattr(overwrites_module.typer, "confirm", lambda *args, **kwargs: True)

    dataset_module.main(Dataset.LENDING_CLUB)

    assert calls == [Dataset.LENDING_CLUB]
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 1
    assert len(parallel_spy.submitted) == 1


def test_main_processes_all_datasets_when_argument_missing(parallel_spy, monkeypatch, tmp_path):
    calls: list[Dataset] = []

    def fake_process(dataset: Dataset):
        calls.append(dataset)
        return (dataset, True, None)

    monkeypatch.setattr(dataset_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(dataset_module, "cpu_count", lambda: 2)
    _mock_processed_path(monkeypatch, tmp_path)
    monkeypatch.setattr(overwrites_module.typer, "confirm", lambda *args, **kwargs: True)

    dataset_module.main(dataset=None)

    assert calls == list(Dataset)
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 2
    assert len(parallel_spy.submitted) == len(Dataset)


def test_main_raises_exit_when_any_dataset_fails(monkeypatch, tmp_path):
    def fake_parallel(*args, **kwargs):  # noqa: ARG001
        def runner(tasks):
            list(tasks)  # Exhaust generator to mimic joblib behavior
            return [
                (Dataset.LENDING_CLUB, True, None),
                (Dataset.CORPORATE_CREDIT_RATING, False, "boom"),
                (Dataset.TAIWAN_CREDIT, True, None),
            ]

        return runner

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(dataset_module, "Parallel", fake_parallel)
    monkeypatch.setattr(dataset_module, "delayed", fake_delayed)
    monkeypatch.setattr(dataset_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(dataset_module, "_process_single_dataset", lambda ds: (ds, True, None))
    _mock_processed_path(monkeypatch, tmp_path)
    monkeypatch.setattr(overwrites_module.typer, "confirm", lambda *args, **kwargs: True)

    with pytest.raises(typer.Exit) as excinfo:
        dataset_module.main(dataset=None)

    assert excinfo.value.exit_code == 1


def test_main_skips_dataset_when_user_declines_overwrite(parallel_spy, monkeypatch, tmp_path):
    processed_path = tmp_path / "lending_club.parquet"
    processed_path.touch()

    def fake_path(dataset):
        return processed_path

    monkeypatch.setattr(dataset_module, "get_processed_path", fake_path)
    monkeypatch.setattr(overwrites_module.typer, "confirm", lambda *args, **kwargs: False)

    dataset_module.main(Dataset.LENDING_CLUB)

    assert parallel_spy.constructor.call_count == 0


def test_main_force_overwrites_without_prompt(parallel_spy, monkeypatch, tmp_path):
    calls: list[Dataset] = []

    def fake_process(dataset: Dataset):
        calls.append(dataset)
        return (dataset, True, None)

    monkeypatch.setattr(dataset_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(dataset_module, "cpu_count", lambda: 2)

    def existing_path(dataset):
        path = tmp_path / f"{dataset.value}.parquet"
        path.touch()
        return path

    monkeypatch.setattr(dataset_module, "get_processed_path", existing_path)
    confirm_mock = Mock(return_value=False)
    monkeypatch.setattr(overwrites_module.typer, "confirm", confirm_mock)

    dataset_module.main(dataset=None, force=True)

    assert confirm_mock.call_count == 0
    assert calls == list(Dataset)


def test_main_respects_jobs_override(parallel_spy, monkeypatch, tmp_path):
    calls: list[Dataset] = []

    def fake_process(dataset: Dataset):
        calls.append(dataset)
        return (dataset, True, None)

    monkeypatch.setattr(dataset_module, "_process_single_dataset", fake_process)
    monkeypatch.setattr(dataset_module, "cpu_count", lambda: 8)
    _mock_processed_path(monkeypatch, tmp_path)
    monkeypatch.setattr(overwrites_module.typer, "confirm", lambda *args, **kwargs: True)

    dataset_module.main(dataset=None, jobs=1)

    assert calls == list(Dataset)
    assert parallel_spy.constructor.call_count == 1
    assert parallel_spy.kwargs["n_jobs"] == 1
