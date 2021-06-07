"""These are just some minimal examples to better illustrate tweeteval's API."""
import pytest
import numpy as np
from sklearn.linear_model import SGDClassifier

import tweeteval as twee
from tweeteval import embed


@pytest.mark.parametrize("task", twee.Task)
def test_score(task):
    """Make sure evaluating gold labels against gold labels always returns a score of 1.0."""
    assert twee.score(twee.test_labels(task), task) == 1


def test_embedding(task=twee.Task.emotion):
    """Make sure embedders generate valid embeddings."""
    texts, _ = twee.task_data(task)
    embedder = embed.TfIdfEmbedder()
    embeddings = embedder.fit_transform(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), 50)


@pytest.mark.xfail
def test_eval_task_tfidf(task=twee.Task.stance):
    """Make sure the whole embedding->classifier pipeline works."""
    embedder = embed.TfIdfEmbedder()
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
    pred, score = twee.eval_task(embedder, model, task)
    print(f"{score=}", flush=True)
    assert score >= 0.6


@pytest.mark.xfail
def test_eval_task_trf(task=twee.Task.stance):
    """Make sure the whole embedding->classifier pipeline works."""
    embedder = embed.TransformersEmbedder(model="cardiffnlp/twitter-roberta-base")
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
    pred, score = twee.eval_task(embedder, model, task)
    print(f"{score=}")
    assert score >= 0.6
