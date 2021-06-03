from functools import partial
from sklearn.metrics import f1_score, recall_score

from .resources import labels_and_preds, TASKS


f1_macro = partial(f1_score, average="macro")
recall_macro = partial(recall_score, average="macro")


def f1_mean(true, pred, labels):
    """Macro (mean) F1 of 'favor' and 'against' classes."""
    return f1_score(true, pred, average=None, labels=labels).mean()


TASK_METRICS = {
    "emoji": f1_macro,
    "emotion": f1_macro,
    "hate": f1_macro,
    "irony": lambda t, p: f1_mean(t, p, labels=["1"]),
    "offensive": f1_macro,
    "sentiment": recall_macro,
    "stance": lambda t, p: f1_mean(t, p, labels=["1", "2"]),
}
"""Metric to use for each task."""


def published_results(tasks=TASKS):
    """Results for best model in paper, i.e. RoBERTa re-trained on Twitter."""
    if isinstance(tasks, str):
        tasks = [tasks]

    return {t: TASK_METRICS[t](*labels_and_preds(t)) for t in tasks}
