"""CLI for experiment management."""

import typer

from .data import app as data

# from .analysis import app as analysis
from .experiment import app as experiment
from .models import app as models

app = typer.Typer()
app.add_typer(data, name="data", help="Data ingestion and preprocessing commands.")
# app.add_typer(analysis, name="analyze", help="Results analysis and visualization commands.")
app.add_typer(models, name="models", help="Model management commands.")
app.add_typer(experiment, name="experiment", help="Experiment execution commands.")
