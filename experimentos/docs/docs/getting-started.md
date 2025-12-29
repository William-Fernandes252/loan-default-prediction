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

Process the raw datasets into an intermediate format:
```bash
make process-data
# or
uv run ldp data process
```

### 2. Feature Preparation

Prepare features for modeling:
```bash
make prepare-features
# or
uv run ldp features prepare
```

### 3. Run Training

Execute the training experiments:
```bash
make train
# or
uv run ldp train experiment
```

### 4. Consolidate Results

Consolidate the results from different experiments:
```bash
make consolidate-results
# or
uv run ldp train consolidate
```

### 5. Analyze Results

Generate analysis reports:
```bash
make analyze
# or
uv run ldp analyze all
```

## Development

### Running Tests
```bash
make test
```

### Linting and Formatting
```bash
make lint
make format
```
