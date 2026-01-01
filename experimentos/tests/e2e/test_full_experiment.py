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


@pytest.fixture(scope="module", autouse=True)
def docker_compose():
    """Starts the E2E stack and tears it down after tests."""
    # Start containers
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.e2e.yml", "up", "-d", "--build"], check=True
    )

    # Wait for LocalStack to be ready (healthcheck in compose helps, but explicit wait is safer)
    print("Waiting for stack to stabilize...")
    time.sleep(10)

    yield

    # Cleanup
    subprocess.run(["docker-compose", "-f", "docker-compose.e2e.yml", "down", "-v"], check=True)


def test_full_pipeline_execution(s3_client: S3Client):
    """
    Scenario:
    1. Create S3 Bucket.
    2. Generate raw data inside the App container.
    3. Upload raw data to S3 (simulating ETL ingestion).
    4. Run 'process' CLI command.
    5. Run 'train' CLI command.
    6. Verify artifacts in S3.
    """

    # Prepare S3
    s3_client.create_bucket(Bucket=BUCKET_NAME)

    # Generate Data & Upload to S3
    # We run the generation script inside the app container to ensure env compatibility,
    # then upload it using aws cli or boto3 from *outside* (or inside if configured).
    # Here we simulate an external ETL process putting data into the raw bucket.

    raw_csv_path = "taiwan_credit.csv"

    # Generate locally for the test runner
    from tests.e2e.scripts.generate_dataset import generate_taiwan_credit

    generate_taiwan_credit(raw_csv_path, n_rows=50)

    # Upload to "Raw" zone in S3
    s3_key = "data/raw/taiwan_credit.csv"
    with open(raw_csv_path, "rb") as f:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=f)

    print(f"Uploaded raw data to s3://{BUCKET_NAME}/{s3_key}")
    os.remove(raw_csv_path)  # Cleanup local

    # Run Data Processing CLI
    # This reads from S3 raw -> transforms -> writes to S3 interim
    print("Running data processing...")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.e2e.yml",
            "exec",
            "-T",
            "app",
            "python",
            "-m",
            "experiments.cli",
            "data",
            "process",
            "taiwan_credit",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Data processing failed: {result.stderr}"

    # Validation: Check if interim parquet exists
    interim_objects = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME, Prefix="data/interim/taiwan_credit.parquet"
    )
    assert "Contents" in interim_objects, "Interim parquet file not found in S3"
    print("Interim data verified in S3.")

    # Run Training Experiment CLI
    # This reads from S3 interim -> trains -> writes results to S3 results
    print("Running training experiment...")
    # Using --jobs 1 for deterministic, low-memory test run
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.e2e.yml",
            "exec",
            "-T",
            "app",
            "python",
            "-m",
            "experiments.cli",
            "train",
            "experiment",
            "taiwan_credit",
            "--jobs",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Validation: Check for results directory and a consolidation file (if consolidated)
    results_objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="results/taiwan_credit")
    assert "Contents" in results_objects, "No results found in S3"

    # Check if a model file was saved (optional, depends on your pipeline config)
    # The pipeline usually saves checkpoints.
    files = [obj["Key"] for obj in results_objects["Contents"]]
    assert any("checkpoints" in f for f in files), "No checkpoints generated"

    print("Training results verified in S3.")
