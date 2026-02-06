# AWS Infrastructure

This project includes a fully automated AWS infrastructure for running training experiments at scale using **AWS Batch**. The setup supports both **GPU** and **CPU-only** modes, with **Spot instances** for cost savings and **On-Demand** fallback for availability.

## Overview

The infrastructure provisions:

- **AWS Batch** compute environments (Spot + On-Demand)
- **ECR** repository for Docker images
- **S3** bucket for experiment data, models, and results
- **CloudWatch** log group for job monitoring
- **VPC endpoint** for private S3 traffic
- **IAM roles** for secure, least-privilege access

All resources are managed with **Terraform**, with remote state stored in S3 + DynamoDB for safe collaboration.

### Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          AWS Batch                                  │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │  Job Queue   │───▶│  Spot Compute Env (priority 1)           │   │
│  │              │    │  g4dn.xlarge / c5.xlarge (GPU toggle)    │   │
│  │              │───▶│  On-Demand Compute Env (priority 2)      │   │
│  └──────────────┘    └──────────────────────────────────────────┘   │
│         ▲                                                           │
│         │  submit                                                   │
│  ┌──────┴───────────────────────────────┐                           │
│  │  Job Definitions (one per dataset)   │                           │
│  │  • corporate-credit-rating           │                           │
│  │  • lending-club                      │                           │
│  │  • taiwan-credit                     │                           │
│  └──────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐        ┌───────────┐       ┌──────────────┐
    │   ECR   │        │  S3 Data  │       │  CloudWatch  │
    │  Image  │        │  Bucket   │       │    Logs      │
    └─────────┘        └───────────┘       └──────────────┘
```

## Prerequisites

Before deploying, ensure you have:

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.0
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with credentials
- [Docker](https://docs.docker.com/get-docker/) installed
- [jq](https://jqlang.github.io/jq/download/) (for job submission scripts)
- An AWS account with permissions to create IAM roles, Batch environments, S3 buckets, ECR repositories, VPC endpoints, and CloudWatch log groups

## Directory Structure

```text
terraform/
├── bootstrap/
│   └── main.tf          # Remote state bucket + DynamoDB lock table
└── main.tf              # Main infrastructure definition
```

## Deployment Guide

### Step 1: Bootstrap Remote State

The bootstrap configuration creates the S3 bucket and DynamoDB table used by Terraform to store its state remotely. This only needs to be run **once**.

```bash
make tf-bootstrap
```

This runs `terraform init && terraform apply` inside `terraform/bootstrap/`. After it completes, the output will show the bucket and table names. These are already configured in the main `terraform/main.tf` backend block.

!!! note
    The bootstrap config uses local state intentionally — it's the one exception to remote state, since the remote state infrastructure doesn't exist yet.

### Step 2: Initialize Terraform

```bash
make tf-init
```

### Step 3: Deploy Infrastructure

```bash
# CPU-only mode (default)
make tf-apply

# GPU mode
cd terraform && terraform apply -var="use_gpu=true"
```

### Step 4: Build and Push the Docker Image

```bash
# CPU-only image (default)
make docker-build
make docker-push

# GPU image
make docker-build GPU=true
make docker-push
```

### Step 5: Submit Training Jobs

Submit one job per dataset to the Batch queue:

```bash
make submit-jobs
```

This submits three independent jobs — one for `corporate_credit_rating`, one for `lending_club`, and one for `taiwan_credit` — allowing them to run in parallel across Batch compute instances.

### Full Deploy (Shortcut)

Build, push, and apply infrastructure in one command:

```bash
make deploy
```

## GPU Toggle

The entire infrastructure adapts based on a single `use_gpu` variable. By default, GPU is **disabled**.

| Concern | `use_gpu = false` (default) | `use_gpu = true` |
|---|---|---|
| **Instance types** | `c5.xlarge`, `c5.2xlarge`, `m5.xlarge`, `m5.2xlarge` | `g4dn.xlarge`, `g4dn.2xlarge` |
| **AMI** | ECS-optimized standard | ECS-optimized GPU (NVIDIA drivers) |
| **EBS volume** | 50 GB gp3 | 100 GB gp3 |
| **GPU resource** | Not requested | 1 GPU per job |
| **Job command** | `experiment run --only-dataset <ds>` | `experiment run --use-gpu --only-dataset <ds>` |
| **`LDP_USE_GPU`** | `"false"` | `"true"` |
| **`LDP_N_JOBS`** | `"4"` (all vCPUs) | `"1"` (GPU-bound) |

To switch modes, re-apply Terraform with the variable override:

```bash
# Switch to GPU
cd terraform && terraform apply -var="use_gpu=true"

# Switch back to CPU
cd terraform && terraform apply -var="use_gpu=false"
```

!!! warning
    When switching modes, you must also rebuild and push the Docker image with the matching `GPU` flag to ensure the correct CUDA libraries are installed.

## Resource Configuration

### Variables

All Terraform variables can be overridden via `-var` flags or a `.tfvars` file.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_name` | string | `loan-default-prediction` | Prefix for all resource names |
| `environment` | string | `dev` | Deployment environment (`dev`, `staging`, `prod`) |
| `aws_region` | string | `us-east-1` | AWS region |
| `use_gpu` | bool | `false` | GPU or CPU-only instances |
| `spot_max_vcpus` | number | `16` | Max vCPUs for Spot compute environment |
| `on_demand_max_vcpus` | number | `8` | Max vCPUs for On-Demand fallback |
| `job_vcpus` | string | `"4"` | vCPUs per training job |
| `job_memory_mib` | string | `"16384"` | Memory per training job (MiB) |
| `root_volume_size_gb` | number | `100` | Root EBS volume size (GB) |

### Using a `.tfvars` File

Create a file like `terraform/gpu.tfvars`:

```hcl
use_gpu             = true
spot_max_vcpus      = 32
on_demand_max_vcpus = 16
job_memory_mib      = "32768"
```

Apply with:

```bash
cd terraform && terraform apply -var-file=gpu.tfvars
```

## Cost Optimization

The infrastructure is designed to minimize costs:

- **Spot instances** (priority 1): Up to 90% savings over On-Demand for GPU workloads
- **On-Demand fallback** (priority 2): Ensures jobs complete even during Spot scarcity
- **Scale to zero**: Both compute environments set `min_vcpus = 0` and `desired_vcpus = 0`, so no instances run when idle
- **VPC endpoint for S3**: Eliminates data transfer charges for S3 traffic
- **ECR lifecycle policy**: Keeps only the last 5 images, preventing unbounded storage costs
- **CloudWatch log retention**: 30 days, preventing log storage accumulation

### Estimated Costs

| Resource | GPU Mode | CPU Mode |
|----------|----------|----------|
| `g4dn.xlarge` Spot | ~$0.16/hr | — |
| `c5.xlarge` Spot | — | ~$0.05/hr |
| S3 storage | ~$0.023/GB/month | Same |
| ECR storage | ~$0.10/GB/month | Same |
| **Idle cost** | **$0** | **$0** |

!!! tip
    Since compute environments scale to zero, you only pay for storage when no jobs are running.

## Spot Instance Handling

Spot instances can be interrupted by AWS with a 2-minute warning. The infrastructure handles this with:

1. **Smart retry strategy**: Jobs automatically retry (up to 3 attempts) when terminated due to Spot reclamation (`Host EC2*` reason), but exit immediately on application errors.
2. **On-Demand fallback**: If Spot capacity is unavailable, the job queue routes to the On-Demand compute environment.
3. **Resume support**: The application's `--execution-id` flag allows resuming interrupted experiments from where they left off.

To resume a Spot-interrupted experiment manually:

```bash
aws batch submit-job \
  --job-name "resume-taiwan-credit" \
  --job-queue "loan-default-prediction-queue" \
  --job-definition "loan-default-prediction-taiwan-credit" \
  --container-overrides '{
    "command": ["experiment", "run", "--only-dataset", "taiwan_credit", "--execution-id", "<previous-execution-id>"]
  }'
```

## Security

The infrastructure follows the **principle of least privilege**:

| Role | Purpose | Permissions |
|------|---------|-------------|
| **ECS Instance Role** | EC2 host management | `AmazonEC2ContainerServiceforEC2Role` |
| **Batch Service Role** | Batch orchestration | `AWSBatchServiceRole` |
| **Batch Execution Role** | Pull images + write logs | `AmazonECSTaskExecutionRolePolicy` |
| **Batch Job Role** | Runtime S3 access | `s3:ListBucket`, `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` (scoped to experiment bucket) |

Additional security measures:

- S3 bucket: public access blocked, server-side encryption (AES-256), versioning enabled
- VPC endpoint: S3 traffic stays on AWS backbone (no public internet)
- `force_destroy` on S3/ECR: disabled in `prod` environment

## Monitoring

### CloudWatch Logs

All Batch job output is streamed to CloudWatch under `/aws/batch/loan-default-prediction`. Each dataset gets its own stream prefix:

```text
/aws/batch/loan-default-prediction/corporate_credit_rating/<job-id>
/aws/batch/loan-default-prediction/lending_club/<job-id>
/aws/batch/loan-default-prediction/taiwan_credit/<job-id>
```

View logs via the AWS Console or CLI:

```bash
# List recent log streams
aws logs describe-log-streams \
  --log-group-name "/aws/batch/loan-default-prediction" \
  --order-by LastEventTime --descending \
  --limit 5

# Tail a specific job's logs
aws logs tail "/aws/batch/loan-default-prediction" \
  --filter-pattern "taiwan_credit" --follow
```

### Batch Job Status

```bash
# List running jobs
aws batch list-jobs --job-queue loan-default-prediction-queue --job-status RUNNING

# Describe a specific job
aws batch describe-jobs --jobs <job-id>
```

## Outputs

After `terraform apply`, the following outputs are available:

| Output | Description |
|--------|-------------|
| `ecr_repo_url` | ECR repository URL for `docker push` |
| `s3_bucket_name` | S3 bucket for experiment data |
| `job_queue_name` | Batch job queue name |
| `job_definition_names` | Job definition names, keyed by dataset |
| `job_definition_arns` | Job definition ARNs, keyed by dataset |
| `use_gpu` | Whether GPU mode is active |

Access outputs with:

```bash
cd terraform && terraform output
```

## Teardown

To destroy all infrastructure (except the bootstrap state bucket):

```bash
cd terraform && terraform destroy
```

!!! danger
    This will delete the S3 data bucket and all experiment results stored in it (unless `environment = "prod"`, in which case `force_destroy` is disabled).

To also destroy the bootstrap state bucket:

```bash
cd terraform/bootstrap && terraform destroy
```

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make docker-build` | Build Docker image (set `GPU=true` for GPU support) |
| `make docker-push` | Tag and push image to ECR |
| `make tf-bootstrap` | Bootstrap remote state (run once) |
| `make tf-init` | Initialize Terraform |
| `make tf-plan` | Preview infrastructure changes |
| `make tf-apply` | Apply infrastructure changes |
| `make deploy` | Full pipeline: build → push → apply |
| `make submit-jobs` | Submit all 3 dataset training jobs |
