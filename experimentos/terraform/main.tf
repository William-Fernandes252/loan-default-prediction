################################################################################
# Terraform Configuration & Backend
################################################################################

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state managed by the bootstrap config in terraform/bootstrap/.
  # Run `cd terraform/bootstrap && terraform init && terraform apply` first,
  # then paste the output values here.
  backend "s3" {
    bucket         = "loan-default-prediction-terraform-state-264981922234"
    key            = "loan-default-prediction/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "loan-default-prediction-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

################################################################################
# Variables
################################################################################

variable "project_name" {
  description = "Project identifier used as prefix for all resources."
  type        = string
  default     = "loan-default-prediction"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "spot_max_vcpus" {
  description = "Maximum vCPUs for the Spot compute environment."
  type        = number
  default     = 16
}

variable "on_demand_max_vcpus" {
  description = "Maximum vCPUs for the On-Demand fallback compute environment."
  type        = number
  default     = 8
}

variable "job_vcpus" {
  description = "vCPUs allocated per training job."
  type        = string
  default     = "4"
}

variable "job_memory_mib" {
  description = "Memory (MiB) allocated per training job."
  type        = string
  default     = "16384"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GB for GPU instances (needs space for CUDA + Docker images)."
  type        = number
  default     = 100
}

################################################################################
# Locals
################################################################################

locals {
  # The three datasets in the project – one Batch job definition per dataset.
  datasets = toset([
    "corporate_credit_rating",
    "lending_club",
    "taiwan_credit",
  ])

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  # Environment variables shared by every job definition.
  common_env_vars = [
    { name = "LDP_STORAGE_PROVIDER", value = "s3" },
    { name = "LDP_STORAGE_S3_BUCKET", value = aws_s3_bucket.experiments_data.id },
    { name = "LDP_STORAGE_S3_REGION", value = var.aws_region },
    { name = "LDP_USE_GPU", value = "true" },
    { name = "LDP_N_JOBS", value = "1" },
    { name = "LDP_MODELS_N_JOBS", value = "2" },
    { name = "LDP_DEBUG", value = "false" },
    { name = "LDP_LOCALE", value = "en_US" },
  ]
}

################################################################################
# Data Sources
################################################################################

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_route_tables" "default" {
  vpc_id = data.aws_vpc.default.id
}

# Resolve the latest ECS-optimized GPU AMI automatically.
data "aws_ssm_parameter" "ecs_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
}

################################################################################
# S3 — Experiment Data Bucket
################################################################################

resource "aws_s3_bucket" "experiments_data" {
  bucket        = "${var.project_name}-experiments-${var.environment}-${data.aws_caller_identity.current.account_id}"
  force_destroy = var.environment != "prod"

  tags = local.common_tags
}

resource "aws_s3_bucket_public_access_block" "experiments_data" {
  bucket = aws_s3_bucket.experiments_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "experiments_data" {
  bucket = aws_s3_bucket.experiments_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "experiments_data" {
  bucket = aws_s3_bucket.experiments_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

################################################################################
# ECR — Container Image Repository
################################################################################

resource "aws_ecr_repository" "repo" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = var.environment != "prod"

  tags = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "repo" {
  repository = aws_ecr_repository.repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

################################################################################
# Networking — Security Group & VPC Endpoint
################################################################################

resource "aws_security_group" "batch_sg" {
  name        = "${var.project_name}-batch-sg"
  description = "Allow outbound access for Batch jobs"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All outbound (ECR, CloudWatch, etc.)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# Gateway VPC endpoint for S3 — keeps Batch ↔ S3 traffic on the AWS backbone,
# eliminating NAT/egress costs and reducing attack surface.
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = data.aws_vpc.default.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  route_table_ids = data.aws_route_tables.default.ids

  tags = merge(local.common_tags, { Name = "${var.project_name}-s3-endpoint" })
}

################################################################################
# IAM — ECS Instance Role (for EC2 hosts)
################################################################################

resource "aws_iam_role" "ecs_instance_role" {
  name = "${var.project_name}-ecs-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_instance_role_batch" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs_instance_profile" {
  name = "${var.project_name}-ecs-instance-profile"
  role = aws_iam_role.ecs_instance_role.name
}

################################################################################
# IAM — Batch Service Role
################################################################################

resource "aws_iam_role" "batch_service_role" {
  name = "${var.project_name}-batch-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "batch_service_role" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

################################################################################
# IAM — Batch Execution Role (ECR pull + CloudWatch)
################################################################################

resource "aws_iam_role" "batch_execution_role" {
  name = "${var.project_name}-batch-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "batch_execution_ecr" {
  role       = aws_iam_role.batch_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

################################################################################
# IAM — Batch Job Role (S3 access at runtime)
################################################################################

resource "aws_iam_role" "batch_job_role" {
  name = "${var.project_name}-batch-job-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "batch_job_s3_access" {
  name = "s3-access"
  role = aws_iam_role.batch_job_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
      ]
      Resource = [
        aws_s3_bucket.experiments_data.arn,
        "${aws_s3_bucket.experiments_data.arn}/*",
      ]
    }]
  })
}

################################################################################
# CloudWatch — Log Group
################################################################################

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.project_name}"
  retention_in_days = 30

  tags = local.common_tags
}

################################################################################
# Launch Template — GPU AMI + EBS
################################################################################

resource "aws_launch_template" "gpu" {
  name_prefix = "${var.project_name}-gpu-lt-"

  image_id = data.aws_ssm_parameter.ecs_gpu_ami.value

  # GPU instances need extra disk for CUDA libraries + Docker images.
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = var.root_volume_size_gb
      volume_type = "gp3"
    }
  }

  tags = local.common_tags
}

################################################################################
# Batch — Compute Environments
################################################################################

# Primary: Spot instances — up to 90% savings over On-Demand.
resource "aws_batch_compute_environment" "gpu_spot" {
  name = "${var.project_name}-gpu-spot"
  type = "MANAGED"

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    bid_percentage      = 100

    instance_type = ["g4dn.xlarge", "g4dn.2xlarge"]

    max_vcpus     = var.spot_max_vcpus
    min_vcpus     = 0
    desired_vcpus = 0

    security_group_ids = [aws_security_group.batch_sg.id]
    subnets            = data.aws_subnets.default.ids

    instance_role = aws_iam_instance_profile.ecs_instance_profile.arn

    launch_template {
      launch_template_id = aws_launch_template.gpu.id
      version            = "$Latest"
    }
  }

  service_role = aws_iam_role.batch_service_role.arn
  depends_on   = [aws_iam_role_policy_attachment.batch_service_role]

  tags = local.common_tags
}

# Fallback: On-Demand instances — guaranteed availability when Spot is scarce.
resource "aws_batch_compute_environment" "gpu_on_demand" {
  name = "${var.project_name}-gpu-on-demand"
  type = "MANAGED"

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"

    instance_type = ["g4dn.xlarge", "g4dn.2xlarge"]

    max_vcpus     = var.on_demand_max_vcpus
    min_vcpus     = 0
    desired_vcpus = 0

    security_group_ids = [aws_security_group.batch_sg.id]
    subnets            = data.aws_subnets.default.ids

    instance_role = aws_iam_instance_profile.ecs_instance_profile.arn

    launch_template {
      launch_template_id = aws_launch_template.gpu.id
      version            = "$Latest"
    }
  }

  service_role = aws_iam_role.batch_service_role.arn
  depends_on   = [aws_iam_role_policy_attachment.batch_service_role]

  tags = local.common_tags
}

################################################################################
# Batch — Job Queue (Spot first, On-Demand fallback)
################################################################################

resource "aws_batch_job_queue" "queue" {
  name     = "${var.project_name}-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_spot.arn
  }

  compute_environment_order {
    order               = 2
    compute_environment = aws_batch_compute_environment.gpu_on_demand.arn
  }

  tags = local.common_tags
}

################################################################################
# Batch — Job Definitions (one per dataset)
################################################################################

resource "aws_batch_job_definition" "training" {
  for_each = local.datasets

  name = "${var.project_name}-${replace(each.key, "_", "-")}"
  type = "container"

  # Spot-aware retry: retry on host termination, exit on app errors.
  retry_strategy {
    attempts = 3

    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*"
    }
    evaluate_on_exit {
      action    = "EXIT"
      on_reason = "*"
    }
  }

  timeout {
    attempt_duration_seconds = 14400 # 4 hours
  }

  container_properties = jsonencode({
    image            = "${aws_ecr_repository.repo.repository_url}:latest"
    jobRoleArn       = aws_iam_role.batch_job_role.arn
    executionRoleArn = aws_iam_role.batch_execution_role.arn

    resourceRequirements = [
      { type = "VCPU", value = var.job_vcpus },
      { type = "MEMORY", value = var.job_memory_mib },
      { type = "GPU", value = "1" },
    ]

    # Override the image CMD; ENTRYPOINT is already `python -m experiments.cli`.
    command = ["run", "--only-dataset", each.key, "--use-gpu"]

    environment = local.common_env_vars

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = each.key
      }
    }
  })

  tags = local.common_tags
}

################################################################################
# Outputs
################################################################################

output "ecr_repo_url" {
  description = "ECR repository URL for docker push."
  value       = aws_ecr_repository.repo.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket for experiment data and results."
  value       = aws_s3_bucket.experiments_data.id
}

output "job_queue_name" {
  description = "Batch job queue name."
  value       = aws_batch_job_queue.queue.name
}

output "job_definition_names" {
  description = "Batch job definition names, keyed by dataset."
  value       = { for k, v in aws_batch_job_definition.training : k => v.name }
}

output "job_definition_arns" {
  description = "Batch job definition ARNs, keyed by dataset."
  value       = { for k, v in aws_batch_job_definition.training : k => v.arn }
}