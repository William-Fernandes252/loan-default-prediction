# GitHub Copilot Instructions

## Project Overview
This is a Python machine learning project for loan default prediction, structured as a CLI application. It leverages `scikit-learn`, `imbalanced-learn`, `xgboost`, and `polars`.

## Architecture & Patterns
- **Package Structure**:
  - `experiments/cli/`: CLI entry points using `typer`.
  - `experiments/core/`: Domain logic (data loading, modeling pipelines).
  - `experiments/services/`: Application services (data management, model versioning).
- **Data Handling**:
  - **Memory Mapping**: The project heavily relies on `joblib` memory mapping for efficient data handling. Always use `ExperimentDataManager.feature_context` to handle `X` and `y` arrays to prevent memory issues.
  - **Data Flow**: Raw data -> Interim -> Processed.
- **Experiment Execution**:
  - Experiments are defined by `ModelType` and `Technique` enums.
  - Execution is parallelized using `joblib.Parallel` in `experiments/cli/train.py`.
  - Configuration is passed via a `Context` object.

## Development Workflow
- **Dependency Management**: Uses `uv`. Run `uv sync` to update environment.
- **Automation**: Always check `Makefile` for available commands.
  - `make format`: Auto-format code using `ruff`.
  - `make lint`: Check code quality with `ruff`.
  - `make test`: Run `pytest`.
- **Linting**: Strict adherence to `ruff` rules. Ensure imports are sorted and types are hinted.

## Coding Conventions
- **CLI**: Implement new commands using `typer`.
- **Logging**: Use `loguru` instead of standard `logging`.
- **Typing**: Use strict type hints. Use `typing_extensions.Annotated` for CLI arguments.
- **Paths**: Always use `pathlib.Path` for file system operations. Constants are in `experiments/config.py`.

## Key Files
- `experiments/config.py`: Central configuration (paths, constants).
- `experiments/cli/train.py`: Main entry point for training experiments.
- `experiments/core/modeling/runner.py`: Core logic for running a single experiment task.
- `pyproject.toml`: Project metadata and dependencies.
