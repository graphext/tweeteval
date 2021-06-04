from pathlib import Path

import typer

from .resources import read_labels, task_labels, TASKS
from .eval import published_results, SCORERS


CLI = typer.Typer()


@CLI.command()
def published():
    """Simply prints published result of best model."""
    typer.echo("Published:")
    print(published_results())


@CLI.command()
def predictions(file: Path, task: str):
    """Evaluates predictions in specified file against label for given task.

    E.g.: the following should return a score of 1,0 exactly:

        >> tweeteval predictions tweeteval/resources/datasets/emoji/test_labels.txt emoji
    """
    if task not in TASKS:
        typer.echo(f"Task must be one of: {TASKS}! Got '{task}'.")
        raise typer.Exit(1)

    pred = read_labels(file)
    labels = task_labels(task)
    if len(pred) != len(labels):
        typer.echo(f"Predictions (n={len(pred)}) don't have correct length for selected task (n={len(labels)})!")
        raise typer.Exit(1)

    score = SCORERS[task](labels, pred)
    print(score)
