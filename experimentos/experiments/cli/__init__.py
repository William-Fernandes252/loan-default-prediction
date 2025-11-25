"""CLI entry point for experiments."""

import typer

from .analysis import app as analysis
from .data import app as data
from .features import app as features
from .train import app as train

app = typer.Typer()
app.add_typer(data, name="data", help="Data ingestion and preprocessing commands.")
app.add_typer(features, name="features", help="Feature extraction and processing commands.")
app.add_typer(train, name="train", help="Model training and evaluation commands.")
app.add_typer(analysis, name="analyze", help="Results analysis and visualization commands.")
