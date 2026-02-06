# Architecture

The project follows a **pipeline-based architecture** with a **dependency injection container** for maximum flexibility and testability. Each major operation is implemented as a composable pipeline with clearly defined protocols.

## Dependency Injection

The application uses `dependency-injector` with a centralized `Container` class that wires all services. This allows for easy swapping of implementations (e.g., using a different storage backend) and simplifies testing.

### Core Services

| Service | Responsibility |
|---------|----------------|
| **ExperimentsSettings** | Configuration from environment variables / `.env` files (Pydantic Settings) |
| **ResourceCalculator** | RAM-based calculation of safe parallel job counts |
| **StorageService** | Unified file I/O abstraction (local, S3, GCS) |

### Data Services

| Service | Responsibility |
|---------|----------------|
| **DataRepository** | Low-level data access for datasets |
| **DataManager** | High-level dataset processing using pipelines |
| **StratifiedDataSplitter** | Stratified train/test splitting preserving class proportions |
| **FeatureExtractor** | Feature extraction and transformation |

### Training Services

| Service | Responsibility |
|---------|----------------|
| **TrainingExecutor** | Executes model training with pipeline orchestration |
| **GridSearchTrainer** | Hyperparameter optimization via grid search |
| **UnbalancedLearnerFactory** | Creates classifiers with imbalance handling techniques |
| **SeedGenerator** | Generates reproducible random seeds |

### Model Services

| Service | Responsibility |
|---------|----------------|
| **ModelRepository** | Model persistence and retrieval |
| **ModelVersioningService** | Model versioning, checkpointing, and loading |
| **ModelResultsEvaluator** | Calculates evaluation metrics from predictions |

### Experiment Services

| Service | Responsibility |
|---------|----------------|
| **ExperimentExecutor** | Coordinates full experiment runs across datasets/models |
| **InferenceService** | Runs predictions using trained models |
| **ModelPredictionsRepository** | Stores and retrieves prediction results |

### Analysis Services

| Service | Responsibility |
|---------|----------------|
| **PredictionsAnalyzer** | Orchestrates analysis pipelines and generates reports/visualizations |
| **AnalysisArtifactsRepository** | Manages analysis outputs (figures, tables) |

## Storage Layer

The storage layer provides a unified interface for file operations across different backends:

| Backend | Implementation | URI Scheme |
|---------|----------------|------------|
| **Local Filesystem** | `LocalStorageService` | `file://` |
| **AWS S3** | `S3StorageService` | `s3://` |
| **Google Cloud Storage** | `GCSStorageService` | `gs://` |

Cloud storage services use **composition-based dependency injection** — the boto3/GCS clients are created by the DI container and injected into the storage services.

For detailed configuration options, see the [Configuration](configuration.md) page. For deploying with S3 on AWS, see the [Infrastructure](infrastructure.md) guide.

## Pipeline Architecture

All major operations are implemented as composable pipelines using the `Pipeline` and `PipelineExecutor` classes in `experiments.lib.pipelines`.

### Pipeline Components

| Component | Description |
|-----------|-------------|
| **Pipeline** | Ordered collection of tasks to execute |
| **PipelineExecutor** | Executes pipelines with lifecycle hooks |
| **Task** | Unit of work within a pipeline |
| **Step** | Individual operation within a task |

### Core Pipelines

#### Data Pipeline
Processes raw datasets into the format required for experiments.

- **Stages**: Load raw data → Transform → Validate → Export

#### Training Pipeline
Orchestrates batch model training across multiple datasets and configurations.

- **Stages**: Generate tasks → Load data → Train models → Persist results

#### Predictions Pipeline
Runs inference using trained models.

- **Stages**: Load model → Load data → Generate predictions → Persist

#### Analysis Pipeline
Generates reports and visualizations from experiment results. The `PredictionsAnalyzer` service orchestrates different analysis types:

- **Summary Table**: LaTeX table with aggregated metrics per technique
- **Tradeoff Plot**: Precision vs sensitivity trade-off visualization
- **Stability Plot**: Boxplot showing variance across seeds
- **Imbalance Impact Plot**: Scatter plot showing metric vs imbalance ratio
- **Cost-Sensitive vs Resampling**: Comparison of cost-sensitive and resampling techniques
- **Metrics Heatmap**: Heatmap of metrics across models and techniques

- **Stages**: Load predictions → Compute metrics → Generate artifacts → Export to storage

## GPU Acceleration

The project supports GPU acceleration using NVIDIA RAPIDS cuML for compatible algorithms. When enabled:

1. **Data Processing**: Polars uses the GPU engine for DataFrame operations
2. **Model Training**: cuML implementations replace scikit-learn where available
3. **Automatic Fallback**: CPU implementations are used when GPU versions are unavailable

For requirements and configuration, see the [Configuration](configuration.md#gpu-acceleration) page.
