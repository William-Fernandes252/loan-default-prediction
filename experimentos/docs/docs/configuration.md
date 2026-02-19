# Configuration

The project is configured through environment variables, which can be set directly or via a `.env` file in the project root. All settings use the `LDP_` prefix.

## Storage Settings

The storage layer supports multiple backends. Use `LDP_STORAGE_PROVIDER` to select the backend.

### Provider Selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_STORAGE_PROVIDER` | `local`, `s3`, `gcs` | `local` | Storage backend to use |
| `LDP_STORAGE_CACHE_DIR` | Path | `None` | Local cache directory for cloud storage |

### Local Storage

Used when `LDP_STORAGE_PROVIDER=local` (default).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_STORAGE_BASE_PATH` | Path | Project root | Base path for all file operations |

### AWS S3

Used when `LDP_STORAGE_PROVIDER=s3`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_STORAGE_S3_BUCKET` | string | *required* | S3 bucket name |
| `LDP_STORAGE_S3_PREFIX` | string | `""` | Key prefix for all objects |
| `LDP_STORAGE_S3_REGION` | string | `None` | AWS region (e.g., `us-east-1`) |
| `LDP_STORAGE_S3_ENDPOINT_URL` | string | `None` | Custom endpoint (for MinIO, LocalStack, etc.) |
| `LDP_STORAGE_S3_ACCESS_KEY_ID` | string | `None` | AWS access key ID |
| `LDP_STORAGE_S3_SECRET_ACCESS_KEY` | string | `None` | AWS secret access key |

### Google Cloud Storage

Used when `LDP_STORAGE_PROVIDER=gcs`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_STORAGE_GCS_BUCKET` | string | *required* | GCS bucket name |
| `LDP_STORAGE_GCS_PREFIX` | string | `""` | Path prefix for all objects |
| `LDP_STORAGE_GCS_PROJECT` | string | `None` | GCP project ID |
| `LDP_STORAGE_GCS_CREDENTIALS_FILE` | string | `None` | Path to service account JSON file |

## Experiment Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_SAMPLER_K_NEIGHBORS` | int | `3` | Number of neighbors for over-sampling techniques (SMOTE) |
| `LDP_BAGGING_ESTIMATORS` | int | `10` | Number of estimators for bagging in cost-sensitive classifiers (e.g., MetaCost) |
| `LDP_CV_FOLDS` | int | `5` | Number of cross-validation folds for hyperparameter tuning |
| `LDP_NUM_SEEDS` | int | `30` | Number of random seeds (experiment iterations) |

## Resource Settings

Settings for controlling parallelism and resource usage.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_SAFETY_FACTOR` | float | `3.5` | Memory safety multiplier for automatic job calculation |
| `LDP_N_JOBS` | int | `1` | Default number of parallel jobs |
| `LDP_MODELS_N_JOBS` | int | `1` | Number of parallel jobs for model training |
| `LDP_USE_GPU` | bool | `False` | Enable GPU acceleration (requires cuML) |
| `LDP_SEQUENTIAL` | bool | `False` | Run training pipelines sequentially instead of in parallel |
| `LDP_DEBUG` | bool | `True` | Enable debug mode for verbose logging |

## Internationalization Settings

Settings for controlling the locale used in generated analysis artifacts.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LDP_LOCALE` | string | `pt_BR` | Default locale for analysis artifact generation (`en_US` or `pt_BR`) |

## GPU Acceleration

The project supports GPU acceleration using [NVIDIA RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/), which provides GPU-accelerated implementations of machine learning algorithms compatible with scikit-learn.

### System Requirements

To use GPU acceleration, you need:

- **NVIDIA GPU**: A CUDA-compatible GPU with compute capability 7.0+ (Volta architecture or newer)
- **CUDA Toolkit**: Version 11.2 or later
- **NVIDIA Driver**: Version 450.80.02 or later
- **Linux OS**: cuML is only supported on Linux

### Installation

GPU dependencies are managed as an optional dependency group. Install them with:

```bash
uv sync --group gpu
```

This installs:

- `cuml-cu12`: GPU-accelerated machine learning algorithms
- `polars[gpu]`: GPU-accelerated DataFrame operations

### Usage

Enable GPU acceleration via environment variable or CLI flag:

```bash
# Via environment variable
export LDP_USE_GPU=true
uv run ldp experiment run

# Via CLI flag
uv run ldp experiment run --use-gpu
uv run ldp data process --use-gpu
```

### Supported Operations

When GPU acceleration is enabled, the following operations are accelerated:

- **Data Processing**: DataFrame operations using Polars GPU engine
- **Model Training**: SVM and other algorithms using cuML implementations
- **Cross-Validation**: Parallel fold execution on GPU

!!! note
    Not all algorithms have GPU implementations. When a GPU implementation is not available, the system falls back to the CPU version automatically.

## Example `.env` File

```bash
# Storage configuration
LDP_STORAGE_PROVIDER=local
LDP_STORAGE_BASE_PATH=/data/experiments

# Experiment configuration
LDP_CV_FOLDS=5
LDP_NUM_SEEDS=30

# Resource configuration
LDP_N_JOBS=4
LDP_MODELS_N_JOBS=2
LDP_USE_GPU=false
LDP_SEQUENTIAL=false
LDP_DEBUG=false

# Internationalization
LDP_LOCALE=pt_BR
```

## Example: S3 Configuration

```bash
# S3 storage
LDP_STORAGE_PROVIDER=s3
LDP_STORAGE_S3_BUCKET=my-experiments-bucket
LDP_STORAGE_S3_REGION=us-west-2
LDP_STORAGE_S3_PREFIX=loan-default-prediction/

# Using MinIO or LocalStack
LDP_STORAGE_S3_ENDPOINT_URL=http://localhost:9000
LDP_STORAGE_S3_ACCESS_KEY_ID=minioadmin
LDP_STORAGE_S3_SECRET_ACCESS_KEY=minioadmin
```

## Example: AWS Batch Deployment

When running on AWS Batch, environment variables are set automatically by the Terraform infrastructure. The job definitions configure:

```bash
LDP_STORAGE_PROVIDER=s3
LDP_STORAGE_S3_BUCKET=<auto-created bucket>
LDP_STORAGE_S3_REGION=us-east-1
LDP_USE_GPU=false  # or true, based on use_gpu Terraform variable
LDP_N_JOBS=4       # or 1 for GPU mode
LDP_MODELS_N_JOBS=2
LDP_SEQUENTIAL=false  # can be overridden via SEQUENTIAL parameter
LDP_DEBUG=false
LDP_LOCALE=en_US
```

S3 authentication uses the IAM job role attached to the Batch job definition — no explicit credentials are needed. See the [Infrastructure](infrastructure.md) guide for the full setup.
