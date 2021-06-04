from pathlib import Path
from functools import partial

THISDIR = Path(__file__).parent.absolute()

TASKS = ["emoji", "emotion", "hate", "irony", "offensive", "stance"]  # "sentiment" # pred and label size mismatch
STANCE_TOPICS = ["abortion", "atheism", "climate", "feminist", "hillary"]


def resource_path(to=None):
    """Get the absolute path to a file in this packages /resources folder."""
    path = THISDIR
    if to is not None:
        path /= to
    return path


DATA_DIR = resource_path("datasets")
PRED_DIR = resource_path("predictions")
LABEL_FNM = "{}/test_labels.txt".format
PREDS_FNM = "{}.txt".format
"""Convenient pre-defined paths."""


def read_labels(path):
    """Read a labels file and return a list of labels."""
    return open(path).read().split("\n")[:-1]


def labels_(task, gold=True, pred_dir=PRED_DIR):
    """Returns the gold labels or predicted labels for a given task."""
    pred_dir = Path(pred_dir)

    if task != "stance":
        path = DATA_DIR / LABEL_FNM(task) if gold else pred_dir / PREDS_FNM(task)
        return read_labels(path)

    labels = []
    path = (DATA_DIR if gold else pred_dir) / task
    for topic in STANCE_TOPICS:
        fnm = LABEL_FNM(topic) if gold else PREDS_FNM(topic)
        labels.extend(read_labels(path / fnm))
    return labels


task_labels = partial(labels_, gold=True)
task_preds = partial(labels_, gold=False)
