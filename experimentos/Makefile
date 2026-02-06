#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = loan-default-prediction
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
AWS_REGION = us-east-1
ECR_REPO_URL = $(shell cd terraform && terraform output -raw ecr_repo_url 2>/dev/null)
IMAGE_TAG ?= latest


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

## Build GPU Docker image
.PHONY: docker-build
docker-build:
	docker build --build-arg GPU=1 -t $(PROJECT_NAME):$(IMAGE_TAG) .

## Tag and push Docker image to ECR
.PHONY: docker-push
docker-push:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPO_URL)
	docker tag $(PROJECT_NAME):$(IMAGE_TAG) $(ECR_REPO_URL):$(IMAGE_TAG)
	docker push $(ECR_REPO_URL):$(IMAGE_TAG)

## Bootstrap Terraform remote state (run once)
.PHONY: tf-bootstrap
tf-bootstrap:
	cd terraform/bootstrap && terraform init && terraform apply

## Initialize Terraform
.PHONY: tf-init
tf-init:
	cd terraform && terraform init

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

## Submit all training jobs (one per dataset)
.PHONY: submit-jobs
submit-jobs:
	@JOB_QUEUE=$$(cd terraform && terraform output -raw job_queue_name); \
	for dataset in corporate_credit_rating lending_club taiwan_credit; do \
		JOB_DEF=$$(cd terraform && terraform output -json job_definition_arns | jq -r ".\"$$dataset\""); \
		echo "Submitting job for $$dataset ..."; \
		aws batch submit-job \
			--job-name "$(PROJECT_NAME)-$$dataset-$$(date +%Y%m%d-%H%M%S)" \
			--job-queue "$$JOB_QUEUE" \
			--job-definition "$$JOB_DEF" \
			--region $(AWS_REGION); \
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
