import typer

app = typer.Typer()


@app.command("infer")
def predict_with_model(
    model_id: str = typer.Argument(..., help="Identifier of the model to use for prediction."),
    input_path: str = typer.Argument(..., help="Path to the input data for prediction."),
    output_path: str = typer.Argument(..., help="Path to save the prediction results."),
):
    """Runs predictions using a specified trained model."""
    ...
