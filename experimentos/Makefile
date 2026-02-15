#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = loan-default-prediction
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
AWS_REGION = us-east-1
ECR_REPO_URL = $(shell cd terraform && terraform output -raw ecr_repo_url 2>/dev/null)
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff check --fix .

## Format source code with ruff
.PHONY: format
format:
	uv run ruff format .

## Format source code with ruff
.PHONY: type-check
type-check:
	uv run pyrefly check .

## Run tests
.PHONY: test
test:
	uv run pytest tests

## Run unit tests
.PHONY: test-unit
test-unit:
	uv run pytest tests/unit

## Run integration tests
.PHONY: test-integration
test-integration:
	uv run pytest tests/integration

## Run end-to-end tests
.PHONY: test-e2e
test-e2e:
	uv run pytest tests/e2e

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Compile translations
.PHONY: compile-i18n
compile-i18n:
	$(PYTHON_INTERPRETER) compile_i18n.py

## Build documentation
.PHONY: docs-build
docs-build:
	uv run mkdocs build -f docs/mkdocs.yml

## Serve documentation
.PHONY: docs-serve
docs-serve:
	uv run mkdocs serve -f docs/mkdocs.yml


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data
.PHONY: process-data
process-data:
	uv run ldp data process

## Run training experiments
.PHONY: train
train:
	uv run ldp experiment run

## Analyze results
.PHONY: analyze
analyze:
	uv run ldp analyze all


#################################################################################
# INFRASTRUCTURE                                                                #
#################################################################################

## Build Docker image (set GPU=true for GPU support)
GPU ?= false
.PHONY: docker-build
docker-build:
	docker build --build-arg GPU=$(if $(filter true,$(GPU)),1,0) -t $(PROJECT_NAME):$(IMAGE_TAG) .

## Tag and push Docker image to ECR
.PHONY: docker-push
docker-push:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPO_URL)
	docker tag $(PROJECT_NAME):$(IMAGE_TAG) $(ECR_REPO_URL):$(IMAGE_TAG)
	docker tag $(PROJECT_NAME):$(IMAGE_TAG) $(ECR_REPO_URL):latest
	docker push $(ECR_REPO_URL):$(IMAGE_TAG)
	docker push $(ECR_REPO_URL):latest

## Bootstrap Terraform remote state (run once)
.PHONY: tf-bootstrap
tf-bootstrap:
	cd terraform/bootstrap && terraform init && terraform apply

## Initialize Terraform
.PHONY: tf-init
tf-init:
	cd terraform && terraform init -backend-config="bucket=$(shell cd terraform/bootstrap && terraform output -raw state_bucket_name)" -backend-config="key=$(PROJECT_NAME).tfstate" -backend-config="region=$(AWS_REGION)"

## Plan Terraform changes
.PHONY: tf-plan
tf-plan:
	cd terraform && terraform plan

## Apply Terraform changes
.PHONY: tf-apply
tf-apply:
	cd terraform && terraform apply

## Full deploy: build, push, apply infra
.PHONY: deploy
deploy: docker-build docker-push tf-apply

## Upload raw datasets to S3
.PHONY: upload-data
upload-data:
	@S3_BUCKET=$$(cd terraform && terraform output -raw s3_bucket_name); \
	echo "Syncing data/raw/ to s3://$$S3_BUCKET/data/raw/ ..."; \
	aws s3 sync data/raw/ "s3://$$S3_BUCKET/data/raw/" --region $(AWS_REGION)

## Submit a data processing job for a specific dataset (set DATASET=name)
.PHONY: submit-data-job
submit-data-job:
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET variable not set. Usage: make submit-data-job DATASET=taiwan_credit"; \
		exit 1; \
	fi
	@JOB_QUEUE=$$(cd terraform && terraform output -raw job_queue_name); \
	JOB_DEF=$$(cd terraform && terraform output -raw data_job_definition_arn); \
	echo "Submitting data processing job for $(DATASET) ..."; \
	aws batch submit-job \
		--job-name "$(PROJECT_NAME)-data-$(DATASET)-$$(date +%Y%m%d-%H%M%S)" \
		--job-queue "$$JOB_QUEUE" \
		--job-definition "$$JOB_DEF" \
		--parameters '{"dataset_name":"$(DATASET)"}' \
		--region $(AWS_REGION)

## Submit all data processing jobs (one per dataset)
.PHONY: submit-data-jobs
submit-data-jobs:
	@JOB_QUEUE=$$(cd terraform && terraform output -raw job_queue_name); \
	JOB_DEF=$$(cd terraform && terraform output -raw data_job_definition_arn); \
	for dataset in corporate_credit_rating lending_club taiwan_credit; do \
		echo "Submitting data processing job for $$dataset ..."; \
		aws batch submit-job \
			--job-name "$(PROJECT_NAME)-data-$$dataset-$$(date +%Y%m%d-%H%M%S)" \
			--job-queue "$$JOB_QUEUE" \
			--job-definition "$$JOB_DEF" \
			--parameters "{\"dataset_name\":\"$$dataset\"}" \
			--region $(AWS_REGION); \
	done

## Submit an experiment job for a specific dataset (set DATASET=name)
## Optional: SEEDS=<num_seeds> CV_FOLDS=<num_folds> NUM_JOBS=<num_parallel_jobs>
.PHONY: submit-job
submit-job:
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET variable not set. Usage: make submit-job DATASET=taiwan_credit"; \
		exit 1; \
	fi
	@JOB_QUEUE=$$(cd terraform && terraform output -raw job_queue_name); \
	JOB_DEF=$$(cd terraform && terraform output -raw job_definition_arn); \
	OVERRIDES=""; \
	ENV_VARS=""; \
	if [ -n "$(SEEDS)" ]; then \
		ENV_VARS="$$ENV_VARS,{\"name\":\"LDP_NUM_SEEDS\",\"value\":\"$(SEEDS)\"}"; \
	fi; \
	if [ -n "$(CV_FOLDS)" ]; then \
		ENV_VARS="$$ENV_VARS,{\"name\":\"LDP_CV_FOLDS\",\"value\":\"$(CV_FOLDS)\"}"; \
	fi; \
	if [ -n "$(NUM_JOBS)" ]; then \
		ENV_VARS="$$ENV_VARS,{\"name\":\"LDP_N_JOBS\",\"value\":\"$(NUM_JOBS)\"}"; \
	fi; \
	if [ -n "$$ENV_VARS" ]; then \
		ENV_VARS=$${ENV_VARS#,}; \
		OVERRIDES="--container-overrides {\"environment\":[$$ENV_VARS]}"; \
	fi; \
	echo "Submitting experiment job for $(DATASET) ..."; \
	aws batch submit-job \
		--job-name "$(PROJECT_NAME)-$(DATASET)-$$(date +%Y%m%d-%H%M%S)" \
		--job-queue "$$JOB_QUEUE" \
		--job-definition "$$JOB_DEF" \
		--parameters '{"dataset_name":"$(DATASET)"}' \
		$$OVERRIDES \
		--region $(AWS_REGION)

## Submit all experiment jobs (one per dataset)
.PHONY: submit-jobs
submit-jobs:
	@for dataset in corporate_credit_rating lending_club taiwan_credit; do \
		$(MAKE) submit-job DATASET=$$dataset SEEDS=$(SEEDS) CV_FOLDS=$(CV_FOLDS) NUM_JOBS=$(NUM_JOBS); \
	done


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
