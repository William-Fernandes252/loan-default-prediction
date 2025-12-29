# CLI Reference

The project provides a comprehensive CLI tool named `ldp` to manage the entire machine learning lifecycle.

## General Usage

```bash
uv run ldp [COMMAND] [SUBCOMMAND] [OPTIONS]
```

## Commands

### `data`
Manage data processing.

- **`process`**: Convert raw data into interim format.
  ```bash
  uv run ldp data process [DATASET_NAME]
  ```

### `features`
Manage feature extraction.

- **`prepare`**: Extract features and targets from interim data.
  ```bash
  uv run ldp features prepare [DATASET_NAME]
  ```

### `train`
Manage model training.

- **`experiment`**: Train models on the processed datasets.
  ```bash
  uv run ldp train experiment [DATASET_NAME] --jobs [N]
  ```
- **`consolidate`**: Consolidate training results.
  ```bash
  uv run ldp train consolidate
  ```

### `analyze`
Analyze results and generate reports.

- **`all`**: Run all analysis tasks.
  ```bash
  uv run ldp analyze all
  ```

## Datasets

The following datasets are supported:
- `corporate_credit_rating`
- `lending_club`
- `taiwan_credit`

## Options

Most commands support the following options:
- `--help`: Show help message and exit.
- `--jobs`: Number of parallel jobs to run (where applicable).
