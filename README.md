# Loan Default Prediction - Comparing strategies for dealing with data imbalance in credit risk assessment

This repository contains the source code and LaTeX documents for my Bachelor's degree thesis in Computer Science.

## Abstract

Credit risk assessment is essential for financial institutions, enabling responsible lending and loss reduction. With the advancement of machine learning, predictive models have become valuable tools for identifying potential default cases. However, default prediction is challenged by the imbalance in datasets, where default cases are significantly rarer than successful payments. This imbalance leads to biased models that often fail to accurately identify high-risk borrowers.

This project aims to perform a comparative evaluation of techniques for handling data imbalance in the context of credit risk assessment, such as data resampling and cost-sensitive learning.

## Repository Structure

The repository is organized into two main parts: the written thesis and the experimental code.

### 1. Thesis (LaTeX)

The root directory contains the LaTeX source code for the dissertation.

- `dissertacao.tex`: Main LaTeX file.
- `capitulos/`: Contains the chapters of the thesis (Introduction, Theoretical Foundation, Tools, Results, Conclusion).
- `pretextual/` & `postextual/`: Front and back matter elements.

### 2. Experiments (Python)

The `experimentos/` directory contains the Python code used for data processing, model training, and analysis.

- **Frameworks & Libraries**: Python, Polars, Pandas, Imbalanced-learn, Scikit-learn, Matplotlib.
- **Structure**: Follows a Cookiecutter Data Science structure.
- **Key Components**:
  - `experiments/`: Source code for the CLI application.
  - `notebooks/`: Jupyter notebooks for exploratory analysis.
  - `data/`: Directory for raw and processed datasets.

For detailed instructions on how to run the experiments, please refer to the [Experiments README](experimentos/README.md).

## Author

**William Fernandes Dias**
Bachelor's Student in Computer Science.

