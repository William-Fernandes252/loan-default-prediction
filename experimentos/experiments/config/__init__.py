"""Configuration module for experiment paths and settings."""

from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

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
