from pathlib import Path
from typing import List, Optional

import typer

from .resources import PRED_DIR, TASKS
from .eval import score


CLI = typer.Typer()


@CLI.command()
def task(task: str, pred: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=True, resolve_path=True)):
    """Evaluates predictions in specified file against label for given task.

    E.g.: the following should return a score of 1.0 exactly:

    >> tweeteval task emoji tweeteval/resources/datasets/emoji/test_labels.txt

    The pred argument can point to a file or a directory containing a file named after the task:

    >> tweeteval task emoji tweeteval/resources/predictions
    """
    print(score(task, pred))


@CLI.command()
def all(
    pred_dir: Path = typer.Argument(default=PRED_DIR, exists=True, file_okay=False, dir_okay=True, resolve_path=True)
):
    """Scores all predictions given a directory containing prediction files.

    THe directory must have the same structure as the repo's predictions/ folder.
    E.g. the following both produce published results for the best model:

    >> tweeteval all tweeteval/resources/predictions
    >> tweeteval all
    """
    scores = {task: score(task, pred_dir) for task in TASKS}
    print(scores)


@CLI.command()
def tasks(
    pred: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=True, resolve_path=True),
    tasks: Optional[List[str]] = typer.Option(None, "--task", "-t"),
):
    """Evaluate all or a subset of tasks using the --task (-t) option.

    >> tweeteval tasks tweeteval/resources/predictions -t emoji -t emotion
    """
    if not tasks:
        tasks = TASKS
    else:
        if any(t not in TASKS for t in tasks):
            typer.echo(f"Each task must be one of {TASKS}! Got {tasks}.")
            typer.Exit(1)

    if len(tasks) > 1 and not pred.is_dir():
        typer.echo(
            (
                "When selecting multiple tasks, pred must point to a directory "
                "containing the corresponding predition files."
            )
        )

    scores = {task: score(task, pred) for task in tasks}
    print(scores)
