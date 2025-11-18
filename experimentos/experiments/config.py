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

RESULTS_DIR = PROJ_ROOT / "results"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Definition of Costs for Optimization (Grid Search)
# Since we don't have a defined matrix, we test different penalties for class 1 (minority)
COST_GRIDS = [
    None,  # Default weight (1:1)
    "balanced",  # Inverse frequency (sklearn default)
    {0: 1, 1: 5},  # 5x weight for minority class error
    {0: 1, 1: 10},  # 10x weight for minority class error
]

# Configuration for internal cross-validation (for GridSearch)
CV_FOLDS = 5
NUM_SEEDS = 30  # Number of experiment repetitions

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
        """Allows registering a dataset-specific processor function."""

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


class ModelType(enum.Enum):
    """Types of models used in experiments."""

    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    ADA_BOOST = "ada_boost"
    MLP = "mlp"

    def get_params(self) -> dict[str, Any]:
        """Returns model-specific parameters."""
        params: dict[ModelType, dict[str, Any]] = {
            ModelType.SVM: {
                "clf__C": [0.1, 1, 10, 100],
                "clf__kernel": ["rbf"],  # Linear can be too slow for large datasets
                "clf__probability": [True],  # Necessary for some metrics or MetaCost
                # For CS-SVM, the weight will be injected dynamically or via grid here
                # If it is Baseline, class_weight is None.
            },
            ModelType.RANDOM_FOREST: {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 5],
                # Random Forest also accepts class_weight, useful for comparison with CS-SVM
            },
            ModelType.ADA_BOOST: {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.01, 0.1, 1.0],
            },
            ModelType.MLP: {
                "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "clf__activation": ["relu", "tanh"],
                "clf__alpha": [0.0001, 0.01],
                "clf__max_iter": [500],  # Ensure convergence
                "clf__early_stopping": [True],
            },
        }
        return params.get(self, {})


class Technique(enum.Enum):
    """Types of techniques used in experiments."""

    BASELINE = "baseline"
    SMOTE = "smote"
    RANDOM_UNDER_SAMPLING = "random_under_sampling"
    SMOTE_TOMEK = "smote_tomek"
    META_COST = "meta_cost"
    CS_SVM = "cs_svm"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
