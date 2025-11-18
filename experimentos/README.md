# Experiments

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img alt="Cookiecutter Data Science badge" src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Loan default prediction training and analysis.

## Project Organization

```text
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         experimentos and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── experimentos   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes experimentos a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Running training experiments

The training CLI now mirrors the dataset and feature preparation commands, supporting multiple
datasets and joblib-based parallelism for faster experimentation.

- **Datasets:** `corporate_credit_rating`, `lending_club`, `taiwan_credit`.
- **Default:** Omit the dataset argument to train every dataset sequentially or in parallel.
- **Parallel jobs:** Use `--jobs / -j` to set the maximum number of worker processes (defaults to the
    detected CPU count and is clamped to the number of scheduled datasets).

Examples:

```bash
# Train every dataset using as many workers as available CPUs
python -m experiments.modeling.train

# Train just the Lending Club dataset with two workers and a custom results filename
python -m experiments.modeling.train lending_club --jobs 2 --output-name tuning.parquet
```

Make sure `python -m experiments.dataset` and `python -m experiments.features` have been run before
invoking training so the processed feature matrices exist.
