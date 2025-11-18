import enum
from pathlib import Path
from typing import Any, Callable, TypeVar, overload

from dotenv import load_dotenv
from loguru import logger
from polars import DataFrame, datatypes

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

_Processor = TypeVar("_Processor", bound=Callable[[DataFrame], DataFrame])


class Dataset(enum.Enum):
    """Datasets used."""

    __dataset_processors: dict[str, Callable[[DataFrame], DataFrame]] = {}
    """Dictionary of registered dataset processors."""

    __feature_extractors: dict[str, Callable[[DataFrame], DataFrame]] = {}
    """Dictionary of registered feature extractors."""

    """Experiment configuration."""
    CORPORATE_CREDIT_RATING = "corporate_credit_rating"
    LENDING_CLUB = "lending_club"
    TAIWAN_CREDIT = "taiwan_credit"

    def __str__(self) -> str:
        return self.value

    def get_path(self) -> Path:
        """Returns the raw data file path for the dataset."""
        return RAW_DATA_DIR / f"{self.value}.csv"

    def get_extra_params(self) -> dict[str, Any]:
        """Returns extra parameters specific to the dataset, if any."""
        extra_params: dict[Dataset, dict[str, Any]] = {
            Dataset.LENDING_CLUB: {"schema_overrides": {"id": datatypes.Utf8}},
            Dataset.TAIWAN_CREDIT: {"infer_schema_length": None},
        }
        return extra_params.get(self, {})

    @overload
    def register_dataset_processor(
        self,
        processor: Callable[[DataFrame], DataFrame],
    ) -> Callable[[DataFrame], DataFrame]: ...

    @overload
    def register_dataset_processor(
        self,
    ) -> Callable[[Callable[[DataFrame], DataFrame]], Callable[[DataFrame], DataFrame]]: ...

    def register_dataset_processor(
        self,
        processor: Callable[[DataFrame], DataFrame] | None = None,
    ) -> (
        Callable[[Callable[[DataFrame], DataFrame]], Callable[[DataFrame], DataFrame]]
        | Callable[
            [DataFrame],
            DataFrame,
        ]
    ):
        """Permite registrar processadores via chamada direta ou com sintaxe de decorator."""

        def decorator(func: _Processor) -> _Processor:
            self.__dataset_processors[self.value] = func
            return func

        if processor is not None:
            return decorator(processor)

        return decorator

    def process_data(self, raw_data: DataFrame) -> DataFrame:
        """Processes raw data using the registered processor for the dataset."""
        processor = self.__dataset_processors.get(self.value)
        if processor is None:
            raise ValueError(f"No processor registered for dataset {self.value}")
        return processor(raw_data)

    def extract_features(self, processed_data: DataFrame) -> DataFrame:
        """Extracts features from processed data using the registered extractor for the dataset."""
        extractor = self.__feature_extractors.get(self.value)
        if extractor is None:
            raise ValueError(f"No feature extractor registered for dataset {self.value}")
        return extractor(processed_data)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
