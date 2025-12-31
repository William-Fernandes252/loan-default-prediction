import pytest

from experiments.core.choices import Choice
from experiments.core.modeling.types import ModelType, Technique


def test_model_type_from_id_and_display_name() -> None:
    assert ModelType.from_id("svm") is ModelType.SVM
    assert ModelType.display_name_from_id("svm") == "SVM"


def test_model_type_missing_accepts_choice_and_string() -> None:
    assert ModelType._missing_(Choice("svm", "SVM")) is ModelType.SVM
    assert ModelType._missing_("RANDOM_FOREST") is ModelType.RANDOM_FOREST


def test_model_type_from_id_invalid_raises() -> None:
    with pytest.raises(ValueError):
        ModelType.from_id("unknown")


def test_technique_from_id_and_display_name() -> None:
    assert Technique.from_id("smote") is Technique.SMOTE
    assert Technique.display_name_from_id("smote_tomek") == "SMOTE Tomek"


def test_technique_missing_accepts_choice_and_string() -> None:
    assert Technique._missing_(Choice("cs_svm", "CSSVM")) is Technique.CS_SVM
    assert Technique._missing_("BASELINE") is Technique.BASELINE


def test_technique_from_id_invalid_raises() -> None:
    with pytest.raises(ValueError):
        Technique.from_id("invalid")
