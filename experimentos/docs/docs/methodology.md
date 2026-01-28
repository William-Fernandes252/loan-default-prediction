# Methodology

This page describes the experimental setup, datasets, and techniques used in this project.

## Datasets

The experiments are conducted on three real-world datasets with varying degrees of imbalance:

| Dataset | Observations | Default Rate | Description |
|---------|--------------|--------------|-------------|
| **Lending Club** | ~630,000 | ~1.1% | Peer-to-peer lending data from the USA. |
| **Taiwan Credit** | ~30,000 | ~22.1% | Credit card default data from Taiwan. |
| **Corporate Credit** | ~2,000 | ~0.05% | Corporate credit ratings and financial ratios. |

## Algorithms

We evaluate the following machine learning algorithms:

- **Support Vector Machine (SVM)**: With RBF kernel.
- **Random Forest**: Ensemble of decision trees.
- **XGBoost**: Gradient boosted decision trees.
- **Multi-layer Perceptron (MLP)**: Feed-forward neural network.

## Imbalance Handling Techniques

The project compares two main approaches:

### 1. Resampling Methods
- **Random Under-sampling (RUS)**: Removes majority class samples.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic minority samples.
- **SMOTE-Tomek**: Hybrid method combining SMOTE with Tomek links for cleaning.

### 2. Cost-Sensitive Learning
- **MetaCost**: A wrapper that makes any classifier cost-sensitive by relabeling training data.
- **Cost-Sensitive SVM (CSSVM)**: Modifies the SVM objective function to penalize misclassifications of the minority class more heavily.

## Experimental Design

To ensure statistical significance and robustness:

- **30 Iterations**: Each experiment is executed 30 times with different random seeds.
- **Stratified Split**: 70% training and 30% testing, preserving class proportions.
- **Hyperparameter Optimization**: 5-fold cross-validation using Grid Search on the training set.
- **Metrics**: We focus on metrics that are robust to imbalance:
    - **G-mean** (Geometric Mean of Sensitivity and Specificity)
    - **Balanced Accuracy**
    - **Recall** (Sensitivity)
    - **F1-Score**

## Mapping Methodology to Implementation

The following table maps the methodological steps to the corresponding CLI commands and code modules:

| Step | CLI Command | Core Module |
|------|-------------|-------------|
| **Data Preprocessing** | `uv run ldp data process` | `experiments.core.data` |
| **Experiment Execution** | `uv run ldp experiment run` | `experiments.core.training` |
| **Single Model Training** | `uv run ldp models train` | `experiments.core.modeling` |
| **Model Predictions** | `uv run ldp models predict` | `experiments.core.predictions` |
| **Results Analysis** | `uv run ldp analyze all` | `experiments.core.analysis` |

## Results and Reports

After running the experiments, results are stored in the `results/` directory. Detailed analysis reports (including LaTeX tables and figures for the thesis) can be generated using the analysis pipeline and are located in:

- `reports/pt_BR/` (Portuguese)
- `reports/en_US/` (English)
