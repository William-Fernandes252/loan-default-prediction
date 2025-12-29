# Architecture

The project follows a **pipeline-based architecture** with a **dependency injection container** for maximum flexibility and testability. Each major operation is implemented as a composable pipeline with clearly defined protocols.

## Dependency Injection

The application uses `dependency-injector` with a centralized `Container` class that wires all services. This allows for easy swapping of implementations (e.g., using a different storage backend) and simplifies testing.

### Core Services

| Service | Responsibility |
|---------|----------------|
| **ExperimentsSettings** | Configuration from environment variables / `.env` files (Pydantic Settings) |
| **PathManager** | Centralized path resolution for data, models, and results |
| **ResourceCalculator** | RAM-based calculation of safe parallel job counts |
| **ModelVersioningServiceFactory** | Factory for model versioning services |
| **ExperimentDataManager** | Dataset artifact management and consolidation |
| **StorageService** | Unified file I/O abstraction (local, S3, GCS) |
| **StorageManager** | High-level storage operations for data and checkpoints |

## Storage Layer

The storage layer provides a unified interface for file operations across different backends:

| Backend | Implementation | URI Scheme |
|---------|----------------|------------|
| **Local Filesystem** | `LocalStorageService` | `file://` |
| **AWS S3** | `S3StorageService` | `s3://` |
| **Google Cloud Storage** | `GCSStorageService` | `gs://` |

Cloud storage services use **composition-based dependency injection** — the boto3/GCS clients are created by the DI container and injected into the storage services.

## Core Pipelines

The project is organized around three main pipelines:

### 1. Training Pipeline
Orchestrates batch model training across multiple datasets and configurations.
- **Stages**: Generate tasks → Load data → Execute → Consolidate

### 2. Experiment Pipeline
Runs a single model experiment, including data splitting, training, and evaluation.
- **Stages**: Split data → Train model → Evaluate → Persist results

### 3. Analysis Pipeline
Generates reports and visualizations from experiment results.
- **Stages**: Load results → Transform data → Export reports
