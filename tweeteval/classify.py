from sklearn.pipeline import make_pipeline
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from transformers import pipeline
from tqdm.auto import tqdm

from .resources import Task, map_labels
from .utils import preprocess


CARDIFF_URL = "cardiffnlp/twitter-roberta-base-{}".format


class PretrainedCardiffClassifier(ClassifierMixin):
    """A minimal sklearn-compmatible wrapper for pretrained CardiffNLP models."""

    def __init__(self, task: Task, humanize=False):
        self.task = task
        self.humanize = humanize
        self.pipe = pipeline("sentiment-analysis", model=CARDIFF_URL(task.name))

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        labels, scores = [], []
        for txt in tqdm(X, desc="Classifying"):
            txt = preprocess(txt)
            res = self.pipe(txt)[0]
            labels.append(res["label"])
            scores.append(res["score"])

        self.scores_ = scores
        labels = [label.replace("LABEL_", "") for label in labels]
        if self.humanize:
            labels = map_labels(labels, task=self.task)
        return labels


def TfidfLogreg(tfidf_cfg=None, svd_cfg=None, lr_cfg=None):
    """TfIdf followed by LogReg, just a template for dumb baselines."""
    tfidf_cfg = tfidf_cfg or {}
    svd_cfg = {**{"n_components": 300}, **(svd_cfg or {})}
    lr_cfg = {**{"Cs": 10, "max_iter": 500}, **(lr_cfg or {})}
    preproc = FunctionTransformer(lambda X: [preprocess(t) for t in X])
    return make_pipeline(
        preproc,
        TfidfVectorizer(**tfidf_cfg),
        TruncatedSVD(**svd_cfg),
        LogisticRegressionCV(**lr_cfg),
    )
