from experiments.core.choices import Choice


def test_choice_str_and_repr_use_id() -> None:
    choice = Choice("test_id", "Display")

    assert str(choice) == "test_id"
    assert repr(choice) == "test_id"
