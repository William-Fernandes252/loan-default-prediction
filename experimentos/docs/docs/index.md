# Loan Default Prediction Experiments

Welcome to the documentation for the Loan Default Prediction experiments project. This project implements the experimental phase of a Bachelor's thesis in Computer Science, focusing on strategies for dealing with data imbalance in credit risk assessment.

## Project Overview

The core problem addressed is the bias in predictive models caused by the rarity of default events in financial datasets. This project evaluates how different resampling and cost-sensitive techniques impact the performance of machine learning models.

### Thesis Structure

The project is documented in the following chapters (available in the root directory as LaTeX files):

1. **Introduction**: Contextualization, problem statement, and objectives.
2. **Theoretical Foundation**: Credit risk, machine learning, and imbalanced learning.
3. **Tools and Methodology**: Description of the datasets, algorithms, and experimental setup.
4. **Results and Discussion**: Analysis of the performance of each strategy.
5. **Conclusion**: Final considerations and future work.

### Key Features

- **Pipeline Architecture**: Modular pipelines for data processing, training, and analysis.
- **Dependency Injection**: Uses `dependency-injector` for flexible service management.
- **Unified Storage**: Abstracted storage layer supporting local filesystem, AWS S3, and Google Cloud Storage.
- **CLI Driven**: Comprehensive command-line interface for managing the entire lifecycle.
- **GPU Acceleration**: Optional GPU support via NVIDIA RAPIDS cuML for faster training.
- **Reproducibility**: Versioned models and datasets with fixed seeds for 30-run iterations.

## Getting Started

To get started with the project, please refer to the [Getting Started](getting-started.md) guide.

## Methodology

For a detailed look at the datasets, algorithms, and experimental design, see the [Methodology](methodology.md) page.

## Architecture

For a detailed look at the project's architecture, see the [Architecture](architecture.md) page.

## CLI Reference

Detailed information about the available CLI commands can be found in the [CLI Reference](cli.md).

## AWS Infrastructure

For deploying experiments at scale on AWS Batch, see the [Infrastructure](infrastructure.md) guide.
