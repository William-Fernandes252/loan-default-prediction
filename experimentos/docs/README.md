# Documentation

This directory contains the project documentation built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## Structure

- `mkdocs.yml`: Configuration file for MkDocs.
- `docs/`: Directory containing the Markdown source files.
  - `index.md`: Home page.
  - `getting-started.md`: Setup and execution guide.
  - `architecture.md`: Project architecture and design.
  - `cli.md`: CLI reference.

## Commands

You can use the following `make` commands from the root directory to manage the documentation:

- **Build documentation**:
  ```bash
  make docs-build
  ```
  This will generate the static site in the `site/` directory.

- **Serve documentation**:
  ```bash
  make docs-serve
  ```
  This will start a local development server (usually at http://127.0.0.1:8000) with live-reloading.

Alternatively, you can use `uv` directly:

```bash
uv run mkdocs build -f docs/mkdocs.yml
uv run mkdocs serve -f docs/mkdocs.yml
```
