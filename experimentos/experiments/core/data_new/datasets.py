import enum
from typing import Any

from polars import datatypes

from experiments.core.choices import Choice


class Dataset(enum.Enum):
    """Datasets used in the experiments."""

    CORPORATE_CREDIT_RATING = Choice("corporate_credit_rating", "Corporate Credit Rating")
    LENDING_CLUB = Choice("lending_club", "Lending Club")
    TAIWAN_CREDIT = Choice("taiwan_credit", "Taiwan Credit")

    def __str__(self) -> str:
        return self.id

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def display_name(self) -> str:
        return self.value.display_name

    @classmethod
    def from_id(cls, dataset_id: str) -> "Dataset":
        for member in cls:
            if member.id == dataset_id:
                return member
        raise ValueError(f"Unknown dataset id: {dataset_id}")

    @classmethod
    def display_name_from_id(cls, dataset_id: str) -> str:
        dataset = cls.from_id(dataset_id)
        return dataset.display_name

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

    def get_extra_params(self) -> dict[str, Any]:
        """Returns extra parameters for parsing the dataset using Polars, if any."""
        extra_params: dict[Dataset, dict[str, Any]] = {
            Dataset.LENDING_CLUB: {"schema_overrides": {"id": datatypes.Utf8}},
            Dataset.TAIWAN_CREDIT: {"infer_schema_length": None},
        }
        return extra_params.get(self, {})
