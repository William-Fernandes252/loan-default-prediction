"""End-to-end tests for the full experiment pipeline.

These tests use Docker Compose to orchestrate LocalStack S3 and the application
container, validating the complete workflow from data processing to experiment execution.
"""

import os
import subprocess
import time

import boto3
import pytest
from types_boto3_s3 import S3Client

# Config matching docker-compose.e2e.yaml
ENDPOINT_URL = "http://localhost:4566"
BUCKET_NAME = "test-experiments"
AWS_CREDS = {
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "region_name": "us-east-1",
}

# Storage paths matching DataStorageLayout and ModelPredictionsStorageLayout
RAW_DATA_PREFIX = "data/raw/"
INTERIM_DATA_PREFIX = "data/interim/"
PROCESSED_DATA_PREFIX = "data/processed/"
PREDICTIONS_PREFIX = "predictions/"
MODELS_PREFIX = "models/"


@pytest.fixture(scope="module")
def s3_client() -> S3Client:
    """Returns a boto3 client connected to LocalStack."""
    return boto3.client("s3", endpoint_url=ENDPOINT_URL, **AWS_CREDS)


@pytest.fixture(scope="session", autouse=True)
def set_environment_variables(num_seeds: int):
    """Sets environment variables for the tests."""
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_CREDS["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_CREDS["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = AWS_CREDS["region_name"]
    os.environ["LDP_NUM_SEEDS"] = str(num_seeds)
    yield
    del os.environ["AWS_ACCESS_KEY_ID"]
    del os.environ["AWS_SECRET_ACCESS_KEY"]
    del os.environ["AWS_DEFAULT_REGION"]
    del os.environ["LDP_NUM_SEEDS"]


@pytest.fixture(scope="module", autouse=True)
def start_stack(s3_client: S3Client):
    """Starts the E2E stack and tears it down after tests."""
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.e2e.yaml",
            "up",
            "-d",
            "--build",
        ],
        check=True,
    )
    print("Started stack.")

    print("Waiting for LocalStack S3...")
    retries = 30
    for i in range(retries):
        try:
            s3_client.list_buckets()
            print("LocalStack S3 is ready.")
            break
        except Exception:
            if i == retries - 1:
                raise Exception("LocalStack S3 failed to become ready.")
            time.sleep(1)

    yield

    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.e2e.yaml", "down", "-v"],
        check=True,
    )


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Helper to run CLI commands inside the container.

    Args:
        *args: Command line arguments to pass to the CLI.

    Returns:
        CompletedProcess instance with command results.
    """
    # Filter out empty strings from arguments
    filtered_args = [arg for arg in args if arg]

    cmd = [
        "docker",
        "compose",
        "-f",
        "docker-compose.e2e.yaml",
        "exec",
        "-T",
        "app",
        "python",
        "-m",
        "experiments.cli",
        *filtered_args,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CLI command failed: {' '.join(cmd)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    return result


def check_gpu_available() -> bool:
    """Check if GPU is available inside the container."""
    check = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.e2e.yaml",
            "exec",
            "-T",
            "app",
            "nvidia-smi",
        ],
        capture_output=True,
    )
    return check.returncode == 0


@pytest.fixture(scope="module")
def setup_test_data(s3_client: S3Client):
    """Generates and uploads test data once for the entire module.

    This avoids repeating expensive setup for every parameterized test case.
    """
    s3_client.create_bucket(Bucket=BUCKET_NAME)

    raw_csv_path = "taiwan_credit.csv"
    container_path = f"/tmp/{raw_csv_path}"

    try:
        # Generate inside container to avoid host dependency issues
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.e2e.yaml",
                "exec",
                "-T",
                "app",
                "python",
                "/app/scripts/generate_dataset.py",
                container_path,
            ],
            check=True,
        )

        # Copy from container to host
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.e2e.yaml",
                "cp",
                f"app:{container_path}",
                raw_csv_path,
            ],
            check=True,
        )

        # Upload to "Raw" zone in S3
        s3_key = f"{RAW_DATA_PREFIX}taiwan_credit.csv"
        with open(raw_csv_path, "rb") as f:
            s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=f)

        print(f"Uploaded raw data to s3://{BUCKET_NAME}/{s3_key}")

    finally:
        if os.path.exists(raw_csv_path):
            os.remove(raw_csv_path)


class DescribeDataProcessing:
    """Tests for the data processing CLI command."""

    @pytest.mark.usefixtures("setup_test_data")
    def it_processes_dataset_and_creates_interim_parquet(self, s3_client: S3Client):
        """Validates that data processing creates interim parquet files."""
        result = run_cli("data", "process", "taiwan_credit", "--jobs", "1")
        assert result.returncode == 0, f"Data processing failed: {result.stderr}"

        # Validation: Check if interim parquet exists
        interim_objects = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f"{INTERIM_DATA_PREFIX}taiwan_credit.parquet",
        )
        assert "Contents" in interim_objects, "Interim parquet file not found in S3"

        # Check that processed features are created (X and y files)
        processed_objects = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=PROCESSED_DATA_PREFIX,
        )
        assert "Contents" in processed_objects, "Processed data files not found in S3"

        files = [obj["Key"] for obj in processed_objects["Contents"]]
        assert any("taiwan_credit_X.parquet" in f for f in files), "Features file not found"
        assert any("taiwan_credit_y.parquet" in f for f in files), "Target file not found"

    @pytest.mark.usefixtures("setup_test_data")
    def it_supports_gpu_processing_when_available(self, s3_client: S3Client):
        """Validates GPU processing option when GPU is available."""
        if not check_gpu_available():
            pytest.skip("Skipping GPU test: nvidia-smi not found inside container.")

        result = run_cli("data", "process", "taiwan_credit", "--jobs", "1", "--use-gpu")
        assert result.returncode == 0, f"GPU data processing failed: {result.stderr}"


class DescribeExperimentExecution:
    """Tests for the experiment execution CLI command."""

    @pytest.mark.usefixtures("setup_test_data")
    def it_runs_experiment_and_generates_predictions(self, s3_client: S3Client):
        """Validates that experiment execution generates predictions."""
        # First, ensure data is processed
        process_result = run_cli("data", "process", "taiwan_credit", "--jobs", "1")
        assert process_result.returncode == 0, f"Data processing failed: {process_result.stderr}"

        # Run experiment with only random_forest to speed up tests
        experiment_result = run_cli(
            "experiment",
            "run",
            "--only-dataset",
            "taiwan_credit",
            "--jobs",
            "1",
            "--models-jobs",
            "1",
            "--exclude-model",
            "svm",
            "--exclude-model",
            "xgboost",
            "--exclude-model",
            "mlp",
        )
        assert experiment_result.returncode == 0, f"Experiment failed: {experiment_result.stderr}"

        # Validation: Check for predictions in S3
        predictions_objects = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=PREDICTIONS_PREFIX,
        )
        assert "Contents" in predictions_objects, "No predictions found in S3"

        files = [obj["Key"] for obj in predictions_objects["Contents"]]
        # Predictions are stored with execution_id/dataset/model_type/technique/seed_N.parquet
        assert any("taiwan_credit" in f and ".parquet" in f for f in files), (
            "No prediction parquet files generated"
        )

        print("Experiment completed successfully with predictions in S3.")

    @pytest.mark.usefixtures("setup_test_data")
    def it_supports_gpu_experiment_execution_when_available(self, s3_client: S3Client):
        """Validates GPU experiment execution when GPU is available."""
        if not check_gpu_available():
            pytest.skip("Skipping GPU test: nvidia-smi not found inside container.")

        # First, ensure data is processed
        process_result = run_cli("data", "process", "taiwan_credit", "--jobs", "1", "--use-gpu")
        assert process_result.returncode == 0, f"Data processing failed: {process_result.stderr}"

        # Run experiment with GPU and only random_forest to speed up tests
        experiment_result = run_cli(
            "experiment",
            "run",
            "--only-dataset",
            "taiwan_credit",
            "--jobs",
            "1",
            "--use-gpu",
            "--exclude-model",
            "svm",
            "--exclude-model",
            "xgboost",
            "--exclude-model",
            "mlp",
        )
        assert experiment_result.returncode == 0, (
            f"GPU experiment failed: {experiment_result.stderr}"
        )


class DescribeModelTraining:
    """Tests for the model training CLI command."""

    @pytest.mark.usefixtures("setup_test_data")
    def it_trains_single_model_and_saves_to_storage(self, s3_client: S3Client):
        """Validates that single model training saves the model to storage."""
        # First, ensure data is processed
        process_result = run_cli("data", "process", "taiwan_credit", "--jobs", "1")
        assert process_result.returncode == 0, f"Data processing failed: {process_result.stderr}"

        # Train a single model
        train_result = run_cli(
            "models",
            "train",
            "taiwan_credit",
            "random_forest",
            "none",
            "--n-jobs",
            "1",
        )
        assert train_result.returncode == 0, f"Model training failed: {train_result.stderr}"

        # Validation: Check for model in S3
        models_objects = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=MODELS_PREFIX,
        )
        assert "Contents" in models_objects, "No models found in S3"

        files = [obj["Key"] for obj in models_objects["Contents"]]
        # Models are stored as models/{dataset}/{model_type}/{technique}/{model_id}.joblib
        assert any(
            "taiwan_credit" in f and "random_forest" in f and ".joblib" in f for f in files
        ), "Trained model file not found"

        print("Model training completed successfully with model saved in S3.")


class DescribeFullPipeline:
    """Tests for the full end-to-end pipeline execution."""

    @pytest.mark.usefixtures("setup_test_data")
    @pytest.mark.parametrize("use_gpu", [False, True])
    def it_executes_full_pipeline_with_cpu_and_gpu(self, s3_client: S3Client, use_gpu: bool):
        """Runs the full pipeline with different configurations (CPU/GPU)."""
        if use_gpu and not check_gpu_available():
            pytest.skip("Skipping GPU test: nvidia-smi not found inside container.")

        gpu_flag = "--use-gpu" if use_gpu else ""

        # Step 1: Data Processing
        print(f"Running data processing ({'GPU' if use_gpu else 'CPU'})...")
        process_result = run_cli("data", "process", "taiwan_credit", "--jobs", "1", gpu_flag)
        assert process_result.returncode == 0, f"Data processing failed: {process_result.stderr}"

        # Validate interim data
        interim_objects = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f"{INTERIM_DATA_PREFIX}taiwan_credit.parquet",
        )
        assert "Contents" in interim_objects, "Interim parquet file not found in S3"

        # Step 2: Experiment Execution (only random_forest to speed up tests)
        print(f"Running experiment ({'GPU' if use_gpu else 'CPU'})...")
        experiment_result = run_cli(
            "experiment",
            "run",
            "--only-dataset",
            "taiwan_credit",
            "--jobs",
            "1",
            "--models-jobs",
            "1",
            "--exclude-model",
            "svm",
            "--exclude-model",
            "xgboost",
            "--exclude-model",
            "mlp",
            gpu_flag,
        )
        assert experiment_result.returncode == 0, f"Experiment failed: {experiment_result.stderr}"

        # Validate predictions
        predictions_objects = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=PREDICTIONS_PREFIX,
        )
        assert "Contents" in predictions_objects, "No predictions found in S3"

        files = [obj["Key"] for obj in predictions_objects["Contents"]]
        assert any("taiwan_credit" in f for f in files), "No Taiwan Credit predictions generated"

        print(f"Full pipeline completed successfully ({'GPU' if use_gpu else 'CPU'}).")


class DescribeModelInference:
    """Tests for model inference CLI command."""

    @pytest.mark.usefixtures("setup_test_data")
    def it_runs_inference_with_trained_model(self, s3_client: S3Client):
        """Validates that inference can run using a trained model."""
        # First, ensure data is processed
        process_result = run_cli("data", "process", "taiwan_credit", "--jobs", "1")
        assert process_result.returncode == 0, f"Data processing failed: {process_result.stderr}"

        # Train a model
        train_result = run_cli(
            "models",
            "train",
            "taiwan_credit",
            "random_forest",
            "none",
            "--n-jobs",
            "1",
        )
        assert train_result.returncode == 0, f"Model training failed: {train_result.stderr}"

        # Run inference (uses latest model by default)
        predict_result = run_cli(
            "models",
            "predict",
            "taiwan_credit",
        )
        assert predict_result.returncode == 0, f"Inference failed: {predict_result.stderr}"

        # Check output contains predictions header
        assert "prediction" in predict_result.stdout.lower(), "Predictions not in output"

        print("Inference completed successfully.")
