terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

variable "project_name" {
  default = "loan-default-prediction"
}

variable "environment" {
  default = "dev"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_s3_bucket" "experiments_data" {
  bucket = "${var.project_name}-experiments-${var.environment}-${data.aws_caller_identity.current.account_id}"
  force_destroy = true # Be careful with this in production!
}

resource "aws_ecr_repository" "repo" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# Role for the EC2 instances running the containers
resource "aws_iam_role" "ecs_instance_role" {
  name = "${var.project_name}-ecs-instance-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_instance_role_batch" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# Custom policy to allow S3 access from the container
resource "aws_iam_role_policy" "s3_access" {
  name = "s3_access"
  role = aws_iam_role.ecs_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [
          aws_s3_bucket.experiments_data.arn,
          "${aws_s3_bucket.experiments_data.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ecs_instance_profile" {
  name = "${var.project_name}-ecs-instance-profile"
  role = aws_iam_role.ecs_instance_role.name
}

# Role for AWS Batch Service
resource "aws_iam_role" "aws_batch_service_role" {
  name = "${var.project_name}-batch-service-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "aws_batch_service_role" {
  role       = aws_iam_role.aws_batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_batch_compute_environment" "gpu_env" {
  name = "${var.project_name}-gpu-env"
  type = "MANAGED"

  compute_resources {
    type                = "SPOT" # HUGE cost savings
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    bid_percentage      = 100
    
    # g4dn are the most cost-effective GPU instances on AWS
    instance_type = ["g4dn.xlarge", "g4dn.2xlarge"] 
    
    max_vcpus = 16
    min_vcpus = 0 # Scales to zero when not in use
    
    security_group_ids = [aws_security_group.batch_sg.id]
    subnets           = data.aws_subnets.default.ids
    
    instance_role = aws_iam_instance_profile.ecs_instance_profile.arn
  }
  
  service_role = aws_iam_role.aws_batch_service_role.arn
  depends_on   = [aws_iam_role_policy_attachment.aws_batch_service_role]
}

resource "aws_security_group" "batch_sg" {
  name        = "${var.project_name}-batch-sg"
  description = "Allow outbound access for Batch jobs"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_batch_job_queue" "queue" {
  name     = "${var.project_name}-queue"
  state    = "ENABLED"
  priority = 1
  
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_env.arn
  }
}

resource "aws_batch_job_definition" "training_job" {
  name = "${var.project_name}-job-def"
  type = "container"
  
  # Retry strategy for Spot interruptions
  retry_strategy {
    attempts = 3
  }

  container_properties = jsonencode({
    image = "${aws_ecr_repository.repo.repository_url}:latest"
    
    # Resource requirements
    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "16384" }, # 16GB
      { type = "GPU", value = "1" }
    ]
    
    # Environment Variables
    environment = [
      { name = "LDP_STORAGE_PROVIDER", value = "s3" },
      { name = "LDP_STORAGE_S3_BUCKET", value = aws_s3_bucket.experiments_data.id },
      { name = "LDP_STORAGE_S3_REGION", value = "us-east-1" },
      { name = "LDP_RESULTS_DIR", value = "results" },
      { name = "LDP_PROCESSED_DATA_DIR", value = "data/processed" }
    ]
  })
}

# Helpers
data "aws_caller_identity" "current" {}

output "ecr_repo_url" {
  value = aws_ecr_repository.repo.repository_url
}

output "s3_bucket_name" {
  value = aws_s3_bucket.experiments_data.id
}

output "job_queue_name" {
  value = aws_batch_job_queue.queue.name
}

output "job_definition_name" {
  value = aws_batch_job_definition.training_job.name
}