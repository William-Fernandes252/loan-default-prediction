# Loan Default Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img alt="Cookiecutter Data Science badge" src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is part of a Bachelor's thesis in Computer Science focused on **Credit Risk Assessment**. The primary challenge addressed is **data imbalance**, where default cases (minority class) are significantly rarer than non-default cases (majority class).

## Goal

The main objective is to perform a comparative evaluation of techniques for handling data imbalance, specifically:

- **Data Resampling**: Under-sampling (RUS), Over-sampling (SMOTE), and Hybrid methods (SMOTE-Tomek).
- **Cost-Sensitive Learning**: MetaCost and Cost-Sensitive SVM.

The experiments evaluate these techniques across three distinct datasets (Lending Club, Taiwan Credit, and Corporate Credit) using robust metrics like G-mean and Balanced Accuracy.

## Project Organization

```text
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs               <- Documentation files (MkDocs). See [docs/README.md](docs/README.md) for details.
├── experiments        <- Source code for use in this project.
│   ├── __init__.py
│   ├── config.py      <- Application configuration
│   ├── containers.py  <- Dependency injection container (dependency-injector)
│   ├── cli            <- Command Line Interface entry points
│   │   ├── analysis.py    <- Results analysis commands
│   │   ├── data.py        <- Data processing commands
│   │   ├── experiment.py  <- Experiment execution commands
│   │   └── models.py      <- Model training and inference commands
│   ├── config         <- Configuration modules
│   │   ├── logging.py     <- Logging configuration
│   │   └── settings.py    <- Pydantic Settings
│   ├── core           <- Core domain logic (pipeline architecture)
│   │   ├── analysis   <- Results analysis (metrics, evaluation)
│   │   ├── data       <- Data loading and dataset definitions
│   │   ├── modeling   <- Model factories, estimators, and scorers
│   │   ├── predictions <- Prediction generation and storage
│   │   └── training   <- Training orchestration (splitters, trainers)
│   ├── lib            <- Reusable library components
│   │   └── pipelines  <- Pipeline execution framework
│   ├── pipelines      <- Pipeline implementations
│   │   ├── analysis   <- Analysis pipeline and tasks
│   │   ├── data       <- Data processing pipeline
│   │   ├── predictions <- Predictions pipeline
│   │   └── training   <- Training pipeline
│   ├── services       <- Application services
│   │   ├── data_manager.py           <- Dataset processing management
│   │   ├── data_repository.py        <- Data access layer
│   │   ├── experiment_executor.py    <- Experiment orchestration
│   │   ├── grid_search_trainer.py    <- Hyperparameter optimization
│   │   ├── inference_service.py      <- Model inference
│   │   ├── model_repository.py       <- Model persistence
│   │   ├── model_versioning.py       <- Model versioning
│   │   ├── resource_calculator.py    <- RAM-based parallelization
│   │   ├── training_executor.py      <- Training execution
│   │   └── ...                        <- Additional services
│   └── storage        <- Storage abstraction layer
│       ├── interface.py   <- Storage protocol
│       ├── local.py       <- LocalStorage (filesystem)
│       ├── s3.py          <- S3Storage (AWS S3, boto3)
│       └── gcs.py         <- GCSStorage (Google Cloud Storage)
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration file
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
└── results            <- Experiment results and checkpoints
```

## Architecture

The project follows a **pipeline-based architecture** with a **dependency injection container** for maximum flexibility and testability. Each major operation is implemented as a composable pipeline with clearly defined protocols.

### Dependency Injection

The application uses `dependency-injector` with a centralized `Container` class that wires all services:

| Service | Responsibility |
|---------|----------------|
| **LdpSettings** | Configuration from environment variables / `.env` files (Pydantic Settings) |
| **ResourceCalculator** | RAM-based calculation of safe parallel job counts |
| **DataManager** | Dataset processing and management |
| **TrainingExecutor** | Training pipeline execution |
| **ExperimentExecutor** | Full experiment orchestration |
| **ModelVersioner** | Model versioning and persistence |
| **InferenceService** | Model inference and predictions |
| **Storage** | Unified file I/O abstraction (local, S3, GCS) |

CLI commands resolve dependencies directly from the container at runtime.

### Storage Layer

The storage layer provides a unified interface for file operations across different backends:

| Backend | Implementation |
|---------|----------------|
| **Local Filesystem** | `LocalStorage` |
| **AWS S3** | `S3Storage` |
| **Google Cloud Storage** | `GCSStorage` |

Cloud storage services use **composition-based dependency injection** — the boto3/GCS clients are created by the DI container and injected into the storage services. This enables:

- Easy mocking for unit tests
- Centralized credential management
- Swappable storage backends via configuration

The `Storage` protocol provides a unified interface for reading/writing datasets, checkpoints, and model artifacts across all storage backends.

### Core Pipelines

| Pipeline | Purpose | Stages |
|----------|---------|--------|
| **Data** | Processes raw datasets | Load raw → Transform → Validate → Export |
| **Training** | Orchestrates batch model training | Generate tasks → Load data → Train models → Persist |
| **Predictions** | Runs model inference | Load model → Load data → Generate predictions → Persist |
| **Analysis** | Generates reports from results | Load results → Transform data → Export reports |

--------

## Running training experiments

The project uses a CLI application to manage the entire machine learning pipeline, from data processing to model training and analysis.

### 1. Data Processing

Process raw datasets into the format required for experiments.

```bash
# Process all datasets
uv run ldp data process

# Process a specific dataset
uv run ldp data process taiwan_credit
```

### 2. Run Experiments

Execute training experiments across datasets and model configurations.

```bash
# Run all experiments
uv run ldp experiment run

# Run experiments on a specific dataset
uv run ldp experiment run --only-dataset taiwan_credit

# Exclude SVM models (slow to train)
uv run ldp experiment run --exclude-model svm

# Continue an interrupted experiment using its execution ID
uv run ldp experiment run --execution-id <execution-id>
```

### 3. Analysis

Generate analysis reports and visualizations.

```bash
# Run all analysis types (uses latest execution)
uv run ldp analyze all

# Generate specific analysis
uv run ldp analyze summary
uv run ldp analyze stability
uv run ldp analyze comparison
uv run ldp analyze heatmap

# Analyze a specific experiment execution by ID
uv run ldp analyze all taiwan_credit --execution-id <execution-id>
```

**Datasets:** `corporate_credit_rating`, `lending_club`, `taiwan_credit`.

### GPU Acceleration

The project supports GPU acceleration using [NVIDIA RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/). To enable GPU support:

1. Install GPU dependencies:
   ```bash
   uv sync --group gpu
   ```

2. Run commands with the `--use-gpu` flag:
   ```bash
   uv run ldp data process --use-gpu
   uv run ldp experiment run --use-gpu
   ```

**Requirements:**
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 11.2+
- Linux OS

## Development & Code Quality

This project uses `make` to automate common development tasks.

### Tests

Run the test suite using `pytest`:

```bash
# Run all tests
make test

# Run specific scopes
make test-unit
make test-integration
make test-e2e
```

The test suite follows a `Describe<ClassName>` / `it_<behavior>` naming convention for clear, behavior-driven test organization. Tests are organized by scope:

- **Unit Tests**: Located in `tests/unit`, testing individual components in isolation.
- **Integration Tests**: Located in `tests/integration`, testing the interaction between components and pipelines.
- **End-to-End Tests**: Located in `tests/e2e`, testing the complete system flow.

### Linting & Formatting

The project uses `ruff` for both linting and formatting.

- **Check code quality:**

  ```bash
  make lint
  ```

- **Auto-format code:**

  ```bash
  make format
  ```

### Type Checking

Static type checking is enforced using `pyright`.

```bash
make type-check
```
