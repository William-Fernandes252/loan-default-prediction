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

# Skip auto-resume and start a fresh execution
uv run ldp experiment run --skip-resume
```

**Auto-Resume Behavior**: By default, the CLI automatically resumes the latest incomplete execution for the specified datasets. This makes AWS Batch retries effective. Use `--skip-resume` to bypass this and start a new execution, or `--execution-id` to resume a specific execution.

### Sequential Execution

By default, the experiment executor runs training pipelines in parallel (limited by available memory). For memory-constrained environments, you can force sequential execution where pipelines run one at a time:

```bash
# Run pipelines sequentially (one at a time)
uv run ldp experiment run --sequential

# Or via environment variable
export LDP_SEQUENTIAL=true
uv run ldp experiment run

# Via Makefile
make train SEQUENTIAL=true
```

Sequential execution:

- Processes one training pipeline completely before starting the next
- Explicitly frees memory between pipelines (calls `gc.collect()`)
- Reduces peak memory usage at the cost of longer total runtime
- Useful for large datasets that don't fit multiple pipelines in memory
- Model-level parallelism (`--models-jobs`) still works within each pipeline

### Running Experiments with Docker Compose

Use the dedicated compose file for isolated experiment runs with explicit resource controls.

```bash
# Validate compose configuration
make compose-experiment-config

# Run all datasets with memory-safe defaults (sequential + 1 job)
make compose-experiment-run

# Run a single dataset
make compose-experiment-run DATASET=taiwan_credit

# Override search scale/resource settings
make compose-experiment-run DATASET=taiwan_credit N_JOBS=1 MODELS_N_JOBS=1 CV_FOLDS=2 NUM_SEEDS=3 SEQUENTIAL=true

# GPU mode (requires NVIDIA runtime)
make compose-experiment-run DATASET=taiwan_credit GPU_PROFILE=true
```

The compose workflow reads defaults from `.env.experiments`.

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

## AWS Infrastructure

The project includes Terraform-managed infrastructure for running experiments at scale on **AWS Batch**, with support for both GPU and CPU-only modes.

### Quick Start

```bash
# 1. Bootstrap Terraform remote state (once)
make tf-bootstrap

# 2. Deploy infrastructure (CPU-only by default)
make tf-init
make tf-apply

# 3. Build and push the Docker image
make docker-build
make docker-push

# 4. Upload raw datasets to S3
make upload-data

# 5. Submit data processing jobs (transform raw → processed data)
make submit-data-jobs

# 6. Submit training jobs (run experiments on processed data)
make submit-jobs
```

For testing with a single dataset before running the full pipeline:

```bash
# Process one dataset
make submit-data-job DATASET=taiwan_credit

# Once data processing completes, train on that dataset
make submit-job DATASET=taiwan_credit
```

#### Customizing Seeds and CV Folds

You can override the number of random seeds and cross-validation folds on training jobs using the `SEEDS` and `CV_FOLDS` parameters:

```bash
# Submit a job with 50 seeds (instead of default 30)
make submit-job DATASET=taiwan_credit SEEDS=50

# Submit with custom CV folds (instead of default 5)
make submit-job DATASET=taiwan_credit CV_FOLDS=10

# Submit with both parameters
make submit-job DATASET=taiwan_credit SEEDS=100 CV_FOLDS=3

# Submit all datasets with the same custom parameters
make submit-jobs SEEDS=50 CV_FOLDS=10
```

These parameters override the `LDP_NUM_SEEDS` and `LDP_CV_FOLDS` environment variables in the container. If not specified, the defaults from the job definition are used (30 seeds, 5 CV folds).

#### Customizing Parallel Jobs

You can also override the number of parallel jobs using the `NUM_JOBS` parameter:

```bash
# Submit a job with 1 parallel job (useful for memory-constrained datasets)
make submit-job DATASET=taiwan_credit NUM_JOBS=1

# Submit with custom seeds, CV folds, and parallel jobs
make submit-job DATASET=taiwan_credit SEEDS=50 CV_FOLDS=10 NUM_JOBS=2

# Submit all datasets with the same custom parallel job count
make submit-jobs NUM_JOBS=1
```

The `NUM_JOBS` parameter overrides the `LDP_N_JOBS` environment variable, which controls how many parallel data processing tasks can run simultaneously. If not specified, it defaults to the number of vCPUs available on the compute instance.

#### Sequential Execution

For memory-constrained jobs, you can force pipelines to run sequentially (one at a time):

```bash
# Submit a job with sequential execution (reduces memory usage)
make submit-job DATASET=taiwan_credit SEQUENTIAL=true

# Combine with other parameters
make submit-job DATASET=lending_club NUM_JOBS=1 SEQUENTIAL=true

# Submit all datasets with sequential execution
make submit-jobs SEQUENTIAL=true
```

The `SEQUENTIAL` parameter sets the `LDP_SEQUENTIAL` environment variable, causing training pipelines to execute one at a time with explicit memory cleanup between pipelines. This trades runtime for lower peak memory usage.

### GPU Mode

To provision GPU instances (`g4dn`) and build with CUDA support:

```bash
make docker-build GPU=true
cd terraform && terraform apply -var="use_gpu=true"
make docker-push
make submit-jobs
```

### Automatic Execution Resumption

Experiment jobs automatically resume interrupted work, making AWS Batch retries effective:

- **First run**: If no previous execution exists, a new UUID7 execution ID is generated
- **Spot interruption**: If AWS Batch retries the job, it automatically detects and resumes the latest incomplete execution
- **Completion**: Once all combinations are finished, subsequent runs exit immediately (idempotent)
- **Manual override**: Use `--execution-id` to explicitly continue a specific execution

This makes Batch's retry strategy (configured for 3 attempts) effective at handling Spot reclamations.

**Example flow:**

1. Job starts: 100 combinations, completes 30, Spot interrupts
2. Batch retries: Auto-detects execution, skips 30, resumes with 70 remaining
3. Completes all 100: Subsequent retries exit immediately (no wasted compute)

### Infrastructure Resources

| Resource | Purpose |
|----------|--------|
| **AWS Batch** | Managed compute (Spot + On-Demand fallback) |
| **ECR** | Docker image registry |
| **S3** | Experiment data, models, and results |
| **CloudWatch** | Job logging (30-day retention) |
| **VPC Endpoint** | Private S3 traffic (no egress costs) |

Cost is near-zero when idle — both compute environments scale to zero. See the [full infrastructure documentation](docs/docs/infrastructure.md) for details on variables, security, monitoring, and teardown.

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
