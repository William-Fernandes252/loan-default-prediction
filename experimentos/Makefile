#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = loan-default-prediction
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python


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
