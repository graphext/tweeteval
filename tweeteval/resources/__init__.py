from pathlib import Path
from functools import partial
from enum import Enum


class Task(Enum):
    emoji = "emoji"
    emotion = "emotion"
    hate = "hate"
    irony = "irony"
    offensive = "offensive"
    sentiment = "sentiment"  # pred and label size mismatch
    stance = "stance"


class StanceTopic(Enum):
    abortion = "abortion"
    atheism = "atheism"
    climate = "climate"
    feminist = "feminist"
    hillary = "hillary"


THISDIR = Path(__file__).parent.absolute()


def resource_path(to=None):
    """Get the absolute path to a file in this package's /resources folder."""
    path = THISDIR
    if to is not None:
        path /= to
    return path


DATA_DIR = resource_path("datasets")
PRED_DIR = resource_path("predictions")
LABEL_FNM = "{}/test_labels.txt".format
PREDS_FNM = "{}.txt".format
"""Convenient pre-defined paths."""


def readlines(path):
    """Read a labels file and return a list of labels."""
    with open(path) as f:
        return f.read().strip().split("\n")


def label_mapping(task: Task, allow_emoji=True):
    """Create a dictionary mapping the integer strings in data files to human-readable labels."""
    lines = readlines(DATA_DIR / task.name / "mapping.txt")
    val_idx = 1 if (task != Task.emoji or allow_emoji) else 2
    return {tokens[0]: tokens[val_idx] for tokens in (line.split("\t") for line in lines)}


def map_labels(labels, task: Task):
    """Map integer string labels to human-readable labels."""
    mapping = label_mapping(task)
    return [mapping.get(label, label) for label in labels]


def task_data(task: Task, split="train", humanize=False):
    """Get texts and corresponding labels for the given task and split."""
    topics = [t.name for t in StanceTopic] if task == Task.stance else [""]
    texts, labels = [], []
    for topic in topics:
        texts.extend(readlines(DATA_DIR / task.name / topic / f"{split}_text.txt"))
        labels.extend(readlines(DATA_DIR / task.name / topic / f"{split}_labels.txt"))

    if humanize:
        labels = map_labels(labels, task)

    return texts, labels


def labels_(task, gold=True, pred_dir=PRED_DIR, humanize=False):
    """Returns the gold labels or predicted labels for a given task."""
    pred_dir = Path(pred_dir)

    if task != Task.stance:
        path = DATA_DIR / LABEL_FNM(task.name) if gold else pred_dir / PREDS_FNM(task.name)
        return readlines(path)

    labels = []
    path = (DATA_DIR if gold else pred_dir) / task.name
    for topic in StanceTopic:
        fnm = LABEL_FNM(topic.name) if gold else PREDS_FNM(topic.name)
        labels.extend(readlines(path / fnm))

    if humanize:
        labels = map_labels(labels, task)

    return labels


test_labels = partial(labels_, gold=True)
test_preds = partial(labels_, gold=False)
