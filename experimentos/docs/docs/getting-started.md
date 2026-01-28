# Getting Started

Follow these steps to set up the project and run your first experiment.

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- Make (optional, but recommended for convenience)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd experimentos
   ```

2. **Create a virtual environment**:
   ```bash
   make create_environment
   # or
   uv venv --python 3.12
   ```

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   make requirements
   # or
   uv sync
   ```

## Running the Pipeline

The project uses a CLI tool named `ldp`. You can run it using `uv run ldp` or by activating the virtual environment and running `ldp`.

### 1. Data Processing

Process the raw datasets into the format required for experiments:

```bash
make process-data
# or
uv run ldp data process
```

This processes all datasets by default. To process a specific dataset:

```bash
uv run ldp data process taiwan_credit
```

### 2. Run Experiments

Execute the training experiments across all datasets and model configurations:

```bash
make train
# or
uv run ldp experiment run
```

You can filter experiments by dataset, model, or technique:

```bash
# Run only on Taiwan Credit dataset
uv run ldp experiment run --dataset taiwan_credit

# Run only Random Forest experiments
uv run ldp experiment run --model random_forest

# Resume an interrupted experiment
uv run ldp experiment run --resume
```

### 3. Analyze Results

Generate analysis reports and visualizations:

```bash
make analyze
# or
uv run ldp analyze all
```

You can also generate specific analysis types:

```bash
# Generate summary tables
uv run ldp analyze summary

# Generate stability boxplots
uv run ldp analyze stability

# Generate comparison plots
uv run ldp analyze comparison
```

## Development

### Running Tests

You can run the entire test suite or specific scopes:

```bash
# All tests
make test

# Specific scopes
make test-unit
make test-integration
make test-e2e
```

### Linting and Formatting
```bash
make lint
make format
```
