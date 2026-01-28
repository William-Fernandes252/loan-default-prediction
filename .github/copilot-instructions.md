# Copilot Instructions for Loan Default Prediction Project

This repository contains a Bachelor's thesis on credit risk assessment comparing strategies for handling data imbalance. It includes both LaTeX documentation and a Python ML experimentation framework.

## Project Structure

- **Root**: LaTeX thesis files (`dissertacao.tex`, `capitulos/`, `pretextual/`, `postextual/`)
- **`experimentos/`**: Python ML experimentation framework using Cookiecutter Data Science structure

## Python Framework (`experimentos/`)

### Architecture Overview

The project uses a **pipeline-based architecture** with **dependency injection** (`dependency-injector`). Key architectural decisions:

1. **DI Container** ([experiments/containers.py](experimentos/experiments/containers.py)): Centralized service wiring. CLI commands resolve dependencies from `container` singleton at runtime.

2. **Storage Abstraction** ([experiments/storage/interface.py](experimentos/experiments/storage/interface.py)): Unified `Storage` protocol supporting local filesystem, AWS S3, and GCS. Storage backend is configured via `LDP_STORAGE_PROVIDER` environment variable.

3. **Pipeline Framework** ([experiments/lib/pipelines/](experimentos/experiments/lib/pipelines/)): Composable pipelines with `Step`, `Task`, `Pipeline`, and `PipelineExecutor`. Each pipeline has a factory, state, and context pattern.

4. **Four Core Pipelines** in `experiments/pipelines/`:
   - `data/`: Raw data → processed datasets
   - `training/`: Model training with cross-validation
   - `predictions/`: Model inference
   - `analysis/`: Results aggregation and reporting

### Adding New Datasets

Use the decorator pattern in [experiments/core/data/transformers.py](experimentos/experiments/core/data/transformers.py):

```python
@register_transformer(Dataset.YOUR_DATASET)
def your_dataset_transformer(df: pl.DataFrame | pl.LazyFrame, use_gpu: bool = False) -> pl.DataFrame:
    # Transform logic using Polars
    return df.lazy().with_columns(...).collect(engine=get_engine(use_gpu))
```

Add the dataset enum in [experiments/core/data/datasets.py](experimentos/experiments/core/data/datasets.py).

### Configuration

Settings use **Pydantic Settings** with `LDP_` prefix ([experiments/config/settings.py](experimentos/experiments/config/settings.py)):

| Variable               | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `LDP_STORAGE_PROVIDER` | `local`, `s3`, or `gcs`                  |
| `LDP_NUM_SEEDS`        | Number of experiment seeds (default: 30) |
| `LDP_USE_GPU`          | Enable GPU acceleration                  |

### CLI Commands

The CLI is built with Typer. Entry point: `uv run ldp <command>`:

```bash
uv run ldp data process              # Process all datasets
uv run ldp experiment run            # Run training experiments
uv run ldp experiment run --only-dataset taiwan_credit --exclude-model svm
uv run ldp analyze all               # Generate analysis reports
```

### Developer Commands

```bash
# Setup & dependencies
uv sync                              # Install dependencies

# Code quality
make lint                            # Ruff linting with auto-fix
make format                          # Ruff formatting
make type-check                      # Pyrefly type checking

# Tests (use descriptive BDD-style naming)
make test                            # All tests
make test-unit                       # Unit tests only
make test-integration                # Integration tests
```

### Testing Conventions

- **BDD-style naming**: Classes use `Describe*` or `For*`, methods use `it_*` or `should_*`
- **Unit tests**: Mock dependencies using fixtures in `conftest.py`
- **Integration tests**: Use fake implementations (e.g., `FakePredictionsRepository`)

Example from [tests/unit/services/test_feature_extractor.py](experimentos/tests/unit/services/test_feature_extractor.py):

```python
class DescribeExtractFeaturesAndTarget:
    def it_separates_target_from_features(self, feature_extractor): ...
```

### Custom Classifiers

The codebase extends scikit-learn classifiers to handle edge cases in imbalanced learning ([experiments/core/modeling/classifiers.py](experimentos/experiments/core/modeling/classifiers.py)):

- **`RobustXGBClassifier` / `RobustSVC`**: Wrap base classifiers with `_ProbabilityMatrixClassesCorrectionMixin` to ensure `predict_proba()` always returns shape `(N, 2)` even when the model collapses to a single class during cross-validation on highly imbalanced folds.

- **`MetaCostClassifier`**: Implements cost-sensitive learning via bagging probability estimation and risk-based relabeling. Includes safety fallback to `DummyClassifier` when relabeling produces single-class data (prevents AdaBoost crashes).

When adding new classifiers, inherit from the mixin if probability output consistency is needed:

```python
class RobustMyClassifier(_ProbabilityMatrixClassesCorrectionMixin, MyBaseClassifier):
    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return self._ensure_two_classes(probas, self.classes_)
```

### Experiment Continuation

Long-running experiments can be resumed using `--execution-id`:

```bash
# Start an experiment (note the execution ID in logs)
uv run ldp experiment run

# Resume after interruption using the same execution ID
uv run ldp experiment run --execution-id 01912345-6789-7abc-8def-0123456789ab
```

The system tracks completed `(dataset, model, technique, seed)` combinations and skips them on resume.

### Key Libraries

- **Polars** for DataFrame operations (prefer over Pandas)
- **scikit-learn** + **imbalanced-learn** for ML pipelines
- **XGBoost** with `RobustXGBClassifier` wrapper for edge cases
- **loguru** for structured logging

### i18n Support

Reports support localization (`locales/en_US/`, `locales/pt_BR/`). Compile translations: `make compile-i18n`

## LaTeX Thesis

Chapters in `capitulos/`: `introducao.tex`, `fundamentacao-teorica.tex`, `ferramentas.tex`, `resultados.tex`, `conclusao.tex`
