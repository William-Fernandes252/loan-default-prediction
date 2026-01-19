"""CLI entry point for experiments."""

import typer

from .analysis import app as analysis
from .data import app as data
from .models import app as models

app = typer.Typer()
app.add_typer(data, name="data", help="Data ingestion and preprocessing commands.")
app.add_typer(analysis, name="analyze", help="Results analysis and visualization commands.")
app.add_typer(models, name="models", help="Model management commands.")
