import os
import subprocess
import time

import boto3
import pytest
from types_boto3_s3 import S3Client

# Config matching docker-compose
ENDPOINT_URL = "http://localhost:4566"
BUCKET_NAME = "test-experiments"
AWS_CREDS = {
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "region_name": "us-east-1",
}


@pytest.fixture(scope="module")
def s3_client():
    """Returns a boto3 client connected to LocalStack."""
    return boto3.client("s3", endpoint_url=ENDPOINT_URL, **AWS_CREDS)


@pytest.fixture(scope="session", autouse=True)
def set_environment_variables(num_seeds: int):
    """Sets environment variables for the tests."""
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_CREDS["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_CREDS["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = AWS_CREDS["region_name"]
    os.environ["S3_ENDPOINT_URL"] = ENDPOINT_URL
    os.environ["S3_BUCKET_NAME"] = BUCKET_NAME
    os.environ["LDP_NUM_SEEDS"] = str(num_seeds)
    yield
    # Cleanup if necessary
    del os.environ["AWS_ACCESS_KEY_ID"]
    del os.environ["AWS_SECRET_ACCESS_KEY"]
    del os.environ["AWS_DEFAULT_REGION"]
    del os.environ["S3_ENDPOINT_URL"]
    del os.environ["S3_BUCKET_NAME"]


@pytest.fixture(scope="module", autouse=True)
def start_stack(s3_client: S3Client):
    """Starts the E2E stack and tears it down after tests."""
    # Start containers
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

    # Wait for LocalStack to be ready
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

    # Cleanup
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.e2e.yaml", "down", "-v"], check=True
    )


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Helper to run CLI commands inside the container.

    Args:
        args: List of command line arguments to pass to the CLI.

    Returns:
        CompletedProcess instance with command results.
    """
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
        *args,
    ]
    cmd.extend(args)

    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def setup_test_data(s3_client: S3Client):
    """Generates and uploads test data once for the entire module.

    This avoids repeating expensive setup for every parameterized test case.
    """
    # Prepare S3
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
        s3_key = "data/raw/taiwan_credit.csv"
        with open(raw_csv_path, "rb") as f:
            s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=f)

        print(f"Uploaded raw data to s3://{BUCKET_NAME}/{s3_key}")

    finally:
        # Cleanup local file
        if os.path.exists(raw_csv_path):
            os.remove(raw_csv_path)


@pytest.mark.usefixtures("setup_test_data")
@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
        True,
    ],
)
def test_full_pipeline_execution(s3_client: S3Client, use_gpu: bool):
    """Runs the full pipeline with different configurations (CPU/GPU)."""

    # Dynamic check: If GPU is requested, ensure the container sees it.
    if use_gpu:
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
        if check.returncode != 0:
            pytest.skip("Skipping GPU test: nvidia-smi not found inside container.")

    # Run Data Processing CLI
    print("Running data processing...")
    run_cli(["data", "process", "taiwan_credit", "--jobs", "1", "--use-gpu" if use_gpu else ""])

    # Validation: Check if interim parquet exists
    interim_objects = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME, Prefix="data/interim/taiwan_credit.parquet"
    )
    assert "Contents" in interim_objects, "Interim parquet file not found in S3"

    # Run Feature Engineering CLI
    print("Running feature engineering...")
    run_cli(["features", "prepare", "taiwan_credit", "--jobs", "1"])

    # Run Training Experiment CLI
    print("Running training experiment...")
    run_cli(
        ["train", "experiment", "taiwan_credit", "--jobs", "1", "--use-gpu" if use_gpu else ""]
    )

    # Validation: Check for results directory
    results_objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="results/taiwan_credit")
    assert "Contents" in results_objects, "No results found in S3"

    files = [obj["Key"] for obj in results_objects["Contents"]]
    assert any("checkpoints" in f for f in files), "No checkpoints generated"

    print(f"Training results verified in S3 ({'GPU' if use_gpu else 'CPU'}).")
