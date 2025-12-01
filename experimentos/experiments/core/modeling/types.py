import enum

from experiments.core.choices import Choice


class ModelType(enum.Enum):
    """Types of models used in experiments."""

    RANDOM_FOREST = Choice("random_forest", "Random Forest")
    SVM = Choice("svm", "SVM")
    XGBOOST = Choice("xgboost", "XGBoost")
    MLP = Choice("mlp", "MLP")

    def __str__(self) -> str:
        return self.id

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def display_name(self) -> str:
        return self.value.display_name

    @classmethod
    def from_id(cls, identifier: str) -> "ModelType":
        for member in cls:
            if member.id == identifier:
                return member
        raise ValueError(f"Unknown model type id: {identifier}")

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        if isinstance(value, Choice):
            for member in cls:
                if member.value == value:
                    return member
        if isinstance(value, str):
            for member in cls:
                if member.id == value or member.name.lower() == value.lower():
                    return member
        return None


class Technique(enum.Enum):
    """Types of techniques used in experiments."""

    BASELINE = Choice("baseline", "Baseline")
    SMOTE = Choice("smote", "SMOTE")
    RANDOM_UNDER_SAMPLING = Choice("random_under_sampling", "RUS")
    SMOTE_TOMEK = Choice("smote_tomek", "SMOTE Tomek")
    META_COST = Choice("meta_cost", "Meta Cost")
    CS_SVM = Choice("cs_svm", "CSSVM")

    def __str__(self) -> str:
        return self.id

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def display_name(self) -> str:
        return self.value.display_name

    @classmethod
    def from_id(cls, identifier: str) -> "Technique":
        for member in cls:
            if member.id == identifier:
                return member
        raise ValueError(f"Unknown technique id: {identifier}")

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        if isinstance(value, Choice):
            for member in cls:
                if member.value == value:
                    return member
        if isinstance(value, str):
            for member in cls:
                if member.id == value or member.name.lower() == value.lower():
                    return member
        return None
