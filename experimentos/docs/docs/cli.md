# CLI Reference

The project provides a comprehensive CLI tool named `ldp` to manage the entire machine learning lifecycle.

## General Usage

```bash
uv run ldp [COMMAND] [SUBCOMMAND] [OPTIONS]
```

## Command Groups

The CLI is organized into four main command groups:

| Command | Description |
|---------|-------------|
| `data` | Data ingestion and preprocessing |
| `experiment` | Experiment execution |
| `models` | Model management (training and inference) |
| `analyze` | Results analysis and visualization |

---

## `data` — Data Processing

Commands for processing raw datasets into the format required for experiments.

### `data process`

Process one or all datasets from raw to interim format.

```bash
uv run ldp data process [DATASET] [OPTIONS]
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `DATASET` | string (optional) | Dataset to process. If omitted, processes all datasets. |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--jobs`, `-j` | int | auto | Number of parallel workers. Defaults to available CPU count. |
| `--force`, `-f` | flag | false | Overwrite existing processed files without prompting |
| `--use-gpu`, `-g` | flag | false | Enable GPU acceleration |

**Examples:**

```bash
# Process all datasets
uv run ldp data process

# Process a specific dataset
uv run ldp data process taiwan_credit

# Force overwrite existing files
uv run ldp data process --force

# Process with GPU acceleration
uv run ldp data process --use-gpu
```

---

## `experiment` — Experiment Execution

Commands for running training experiments across datasets and model configurations.

### `experiment run`

Run experiments on specified datasets and models.

```bash
uv run ldp experiment run [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--only-dataset` | string | all | Dataset to run experiments on |
| `--exclude-model`, `-x` | string (multiple) | none | Model types to exclude |
| `--jobs`, `-j` | int | auto | Number of parallel jobs (based on RAM) |
| `--models-jobs`, `-m` | int | auto | Number of parallel jobs for model training |
| `--use-gpu`, `-g` | flag | false | Enable GPU acceleration |
| `--execution-id`, `-e` | string | none | Continue a previous experiment by its execution ID |

**Examples:**

```bash
# Run all experiments
uv run ldp experiment run

# Run experiments on a specific dataset
uv run ldp experiment run --only-dataset taiwan_credit

# Exclude SVM models (slow to train)
uv run ldp experiment run --exclude-model svm

# Continue an interrupted experiment
uv run ldp experiment run --execution-id <execution-id>

# Run with GPU acceleration
uv run ldp experiment run --use-gpu
```

---

## `models` — Model Management

Commands for training individual models and running inference.

### `models train`

Train a specific model using the specified dataset, model type, and technique.

```bash
uv run ldp models train DATASET MODEL_TYPE TECHNIQUE [OPTIONS]
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `DATASET` | string | Dataset to train on |
| `MODEL_TYPE` | string | Model type (see [Available Models](#available-models)) |
| `TECHNIQUE` | string | Imbalance technique (see [Available Techniques](#available-techniques)) |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n-jobs`, `-j` | int | 1 | Number of parallel jobs |
| `--use-gpu`, `-g` | flag | false | Enable GPU acceleration |

**Examples:**

```bash
# Train a Random Forest with SMOTE on Taiwan Credit
uv run ldp models train taiwan_credit random_forest smote

# Train XGBoost with cost-sensitive learning
uv run ldp models train lending_club xgboost meta_cost --n-jobs 4
```

### `models predict`

Run inference on a dataset using a trained model.

```bash
uv run ldp models predict DATASET [OPTIONS]
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `DATASET` | string | Dataset to run predictions on |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-id`, `-m` | string | latest | Model ID or version to use |
| `--output`, `-o` | file | stdout | Output file for predictions |

**Examples:**

```bash
# Run predictions using the latest model
uv run ldp models predict taiwan_credit

# Run predictions using a specific model version
uv run ldp models predict taiwan_credit --model-id abc123
```

---

## `analyze` — Results Analysis

Commands for generating analysis reports and visualizations from experiment results.

### `analyze all`

Run all analysis types sequentially.

```bash
uv run ldp analyze all [DATASET] [OPTIONS]
```

### `analyze summary`

Generate a summary table of experiment results in LaTeX format.

```bash
uv run ldp analyze summary [DATASET] [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--technique`, `-t` | string | all | Filter by technique |

### `analyze tradeoff`

Generate a precision-sensitivity trade-off plot.

```bash
uv run ldp analyze tradeoff [DATASET] [OPTIONS]
```

### `analyze stability`

Generate a stability boxplot showing variance across seeds.

```bash
uv run ldp analyze stability [DATASET] [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--metric`, `-m` | string | `balanced_accuracy` | Metric to analyze |

### `analyze imbalance`

Generate an imbalance impact scatter plot.

```bash
uv run ldp analyze imbalance [DATASET] [OPTIONS]
```

### `analyze comparison`

Generate a cost-sensitive vs resampling comparison plot.

```bash
uv run ldp analyze comparison [DATASET] [OPTIONS]
```

### `analyze heatmap`

Generate a metrics heatmap.

```bash
uv run ldp analyze heatmap [DATASET] [OPTIONS]
```

**Common Options for Analysis Commands:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--force`, `-f` | flag | false | Force overwrite of existing artifacts |
| `--gpu` | flag | false | Enable GPU acceleration if available |
| `--locale`, `-l` | string | `pt_BR` | Locale for generated artifacts (`en_US` or `pt_BR`) |

---

## Datasets

The following datasets are supported:

| Dataset ID | Description |
|------------|-------------|
| `corporate_credit_rating` | Corporate credit ratings and financial ratios (~2,000 samples) |
| `lending_club` | Peer-to-peer lending data from the USA (~630,000 samples) |
| `taiwan_credit` | Credit card default data from Taiwan (~30,000 samples) |

---

## Available Models

| Model ID | Description |
|----------|-------------|
| `random_forest` | Random Forest classifier |
| `svm` | Support Vector Machine with RBF kernel |
| `xgboost` | XGBoost gradient boosted trees |
| `mlp` | Multi-layer Perceptron neural network |

---

## Available Techniques

| Technique ID | Description |
|--------------|-------------|
| `baseline` | No imbalance handling |
| `random_under_sampling` | Random Under-Sampling (RUS) |
| `smote` | Synthetic Minority Over-sampling Technique |
| `smote_tomek` | SMOTE + Tomek links hybrid |
| `meta_cost` | MetaCost cost-sensitive wrapper |
| `cs_svm` | Cost-Sensitive SVM |

---

## Available Metrics

| Metric ID | Description |
|-----------|-------------|
| `balanced_accuracy` | Balanced accuracy score |
| `g_mean` | Geometric mean of sensitivity and specificity |
| `f1_score` | F1 score |
| `precision` | Precision |
| `sensitivity` | Sensitivity (Recall) |
| `specificity` | Specificity |

---

## Global Options

All commands support these options:

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
