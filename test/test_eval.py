"""These are just some minimal examples to better illustrate tweeteval's API."""
import pytest
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier

import tweeteval as twee


def TfIdfEmbedder(tfidf_cfg=None, tsvd_cfg=None):
    """Returns a pipeline applying TfIdf->SVD->Normalize."""
    tfidf_cfg = tfidf_cfg or {}
    tsvd_cfg = tsvd_cfg or {"n_components": 50}
    return make_pipeline(TfidfVectorizer(**tfidf_cfg), TruncatedSVD(**tsvd_cfg), Normalizer(copy=False))


@pytest.mark.parametrize("task", twee.Task)
def test_score(task):
    """Make sure evaluating gold labels against gold labels always returns a score of 1.0."""
    assert twee.score(twee.test_labels(task), task) == 1


def test_embedding(task=twee.Task.emotion):
    """Make sure embedders generate valid embeddings."""
    texts, _ = twee.task_data(task)
    embedder = TfIdfEmbedder()
    embeddings = embedder.fit_transform(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), 50)


@pytest.mark.xfail
def test_eval_task(task=twee.Task.stance):
    """Make sure the whole embedding->classifier pipeline works."""
    embedder = TfIdfEmbedder()
    model = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
    pred, score = twee.eval_task(embedder, model, task)
    print(f"{score=}")
    assert score >= 0.6
