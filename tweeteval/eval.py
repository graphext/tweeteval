from pathlib import Path
from functools import partial
from sklearn.metrics import f1_score, recall_score

from .resources import TASKS, read_labels, task_labels, task_preds


f1_macro = partial(f1_score, average="macro")
recall_macro = partial(recall_score, average="macro")


def f1_mean(true, pred, labels):
    """Macro (mean) F1 of selected classes only."""
    return f1_score(true, pred, average=None, labels=labels).mean()


SCORERS = {
    "emoji": f1_macro,
    "emotion": f1_macro,
    "hate": f1_macro,
    "irony": lambda t, p: f1_mean(t, p, labels=["1"]),
    "offensive": f1_macro,
    "sentiment": recall_macro,
    "stance": lambda t, p: f1_mean(t, p, labels=["1", "2"]),
}
"""Metric to use for each task."""


def ensure_labels(pred, task=None):
    """Ensures pred contains labels, reading from file if it points to data on disk."""
    if isinstance(pred, str):
        pred = Path(pred)

    if isinstance(pred, Path):
        if pred.is_dir():
            if task is None:
                raise ValueError("Need a task name to read labels from a directory!")
            pred = task_preds(task=task, pred_dir=pred)
        elif pred.is_file():
            pred = read_labels(pred)
        else:
            raise ValueError(f"{pred} must be a file containing labels, or a directory containing such a file.")

    return pred


def score(task, pred):
    """Return the score for a single task given a predictions file."""
    if task not in TASKS:
        raise ValueError(f"Task must be one of: {TASKS}! Got '{task}'.")

    pred = ensure_labels(pred, task)
    labels = task_labels(task)
    if len(pred) != len(labels):
        raise ValueError(f"Predictions (n={len(pred)}) don't have correct length for selected task (n={len(labels)})!")

    return SCORERS[task](labels, pred)
