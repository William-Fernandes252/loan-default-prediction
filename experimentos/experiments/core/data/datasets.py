import enum
from typing import Any

from polars import datatypes


class Dataset(enum.StrEnum):
    """Datasets used in the experiments."""

    CORPORATE_CREDIT_RATING = "corporate_credit_rating"
    LENDING_CLUB = "lending_club"
    TAIWAN_CREDIT = "taiwan_credit"

    def get_extra_params(self) -> dict[str, Any]:
        """Returns extra parameters for parsing the dataset using Polars, if any."""
        extra_params: dict[Dataset, dict[str, Any]] = {
            Dataset.LENDING_CLUB: {"schema_overrides": {"id": datatypes.Utf8}},
            Dataset.TAIWAN_CREDIT: {"infer_schema_length": None},
        }
        return extra_params.get(self, {})
