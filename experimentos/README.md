# Experiments

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img alt="Cookiecutter Data Science badge" src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Loan default prediction training and analysis.

## Project Organization

```text
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs               <- Documentation files (MkDocs).
├── experiments        <- Source code for use in this project.
│   ├── __init__.py
│   ├── config.py      <- Store useful variables and configuration
│   ├── context.py     <- Context management for the application
│   ├── cli            <- Command Line Interface entry points
│   │   ├── analysis.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── core           <- Core domain logic (pipeline architecture)
│   │   ├── analysis   <- Results analysis pipeline (load → transform → export)
│   │   ├── data       <- Data loading and dataset definitions
│   │   ├── experiment <- Single experiment pipeline (split → train → evaluate → persist)
│   │   ├── modeling   <- Model factories, estimators, and metrics
│   │   └── training   <- Training orchestration pipeline (generate → execute → consolidate)
│   ├── services       <- Application services (DataManager, ModelVersioning)
│   └── utils          <- Utility functions
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration file
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
└── results            <- Experiment results and checkpoints
```

## Architecture

The project follows a **pipeline-based architecture** with dependency injection for maximum flexibility and testability. Each major operation is implemented as a composable pipeline with clearly defined protocols.

### Core Pipelines

| Pipeline | Purpose | Stages |
|----------|---------|--------|
| **Training** | Orchestrates batch model training | Generate tasks → Load data → Execute → Consolidate |
| **Experiment** | Runs a single model experiment | Split data → Train model → Evaluate → Persist results |
| **Analysis** | Generates reports from results | Load results → Transform data → Export reports |

--------

## Running training experiments

The project uses a CLI application to manage the entire machine learning pipeline, from data processing to model training and analysis.

### 1. Data Processing

Convert raw data into interim format.

```bash
python -m experiments.cli data [DATASET_NAME]
```

### 2. Feature Extraction

Extract features and targets from interim data.

```bash
python -m experiments.cli features [DATASET_NAME]
```

### 3. Model Training

Train models on the processed datasets.

```bash
# Train all datasets
python -m experiments.cli train

# Train a specific dataset with parallel jobs
python -m experiments.cli train [DATASET_NAME] --jobs 4
```

### 4. Analysis

Analyze results and generate reports.

```bash
python -m experiments.cli analyze
```

**Datasets:** `corporate_credit_rating`, `lending_club`, `taiwan_credit`.

Make sure to run the steps in order: `data` -> `features` -> `train`.

## Development & Code Quality

This project uses `make` to automate common development tasks.

### Tests

Run the test suite using `pytest`:

```bash
make test
```

The test suite follows a `Describe<ClassName>` / `it_<behavior>` naming convention for clear, behavior-driven test organization.

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

Static type checking is enforced using `mypy`.

```bash
mypy experiments
```
