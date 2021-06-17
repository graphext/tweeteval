from pathlib import Path
from functools import partial
from numbers import Number
from typing import Iterable, Optional

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.metrics import f1_score, recall_score

from .resources import Task, readlines, task_data, test_labels, test_preds
from .utils import preprocess


f1_macro = partial(f1_score, average="macro")
recall_macro = partial(recall_score, average="macro")


def f1_mean(true, pred, labels):
    """Macro (mean) F1 of selected classes only."""
    return f1_score(true, pred, average=None, labels=labels).mean()


SCORERS = {
    Task.emoji: f1_macro,
    Task.emotion: f1_macro,
    Task.hate: f1_macro,
    Task.irony: lambda t, p: f1_mean(t, p, labels=["1"] if "1" in t else ["irony"]),
    Task.offensive: f1_macro,
    Task.sentiment: recall_macro,
    Task.stance: lambda t, p: f1_mean(t, p, labels=["1", "2"] if "1" in t else ["against", "favor"]),
}
"""Metric to use for each task."""


def ensure_labels(pred: Iterable, task: Optional[Task] = None) -> Iterable:
    """Ensures pred contains labels, reading from file if it points to data on disk."""
    if isinstance(pred, str):
        pred = Path(pred)

    if isinstance(pred, Path):
        if pred.is_dir():
            if task is None:
                raise ValueError("Need a task name to read labels from a directory!")
            pred = test_preds(task=task, pred_dir=pred)
        elif pred.is_file():
            pred = readlines(pred)
        else:
            raise ValueError(f"{pred} must be a file containing labels, or a directory containing such a file.")

    return pred


def score(pred: Iterable, task: Task) -> Number:
    """Return the score for a single task given predictions for test split."""
    pred = ensure_labels(pred, task)
    labels = test_labels(task)
    if len(pred) != len(labels):
        raise ValueError(f"Predictions (n={len(pred)}) don't have correct length for selected task (n={len(labels)})!")

    return SCORERS[task](labels, pred)


def eval_classifier(clf: ClassifierMixin, task: Task, preproc=True, embedder: Optional[TransformerMixin] = None):
    """Given sklearn-compatible embedder and classification model, evaluate both on tweeteval task."""
    X_train, y_train = task_data(task, split="train")
    X_test, y_test = task_data(task, split="test")

    if preproc:
        X_train = [preprocess(txt) for txt in X_train]
        X_test = [preprocess(txt) for txt in X_test]

    if embedder:
        X_train = embedder.fit_transform(X_train)
        X_test = embedder.transform(X_test)

    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    s = SCORERS[task](y_test, y_pred)
    return s, y_pred
