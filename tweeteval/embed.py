"""Helpers to use different embeddings with a common sklearn interface."""
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from tqdm.auto import tqdm


def TfIdfEmbedder(tfidf_cfg=None, tsvd_cfg=None):
    """Returns a pipeline applying TfIdf->SVD->Normalize."""
    tfidf_cfg = tfidf_cfg or {}
    tsvd_cfg = tsvd_cfg or {"n_components": 50}
    return make_pipeline(TfidfVectorizer(**tfidf_cfg), TruncatedSVD(**tsvd_cfg), Normalizer(copy=False))


class SpacyEmbedder(BaseEstimator, TransformerMixin):
    """Minimal sklearn-compatible wrapper for spacy embeddings."""

    def __init__(self, model="en_core_web_md", normalize=True):
        self.nlp = spacy.load(model)
        self.normalize = normalize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        docs = self.nlp.pipe(X)
        if self.normalize:
            return np.vstack([d.vector / d.vector_norm for d in docs])
        else:
            return np.vstack([d.vector for d in docs])


class TransformersEmbedder(BaseEstimator, TransformerMixin):
    """Embeds texts using a transformers model."""

    def __init__(self, model="cardiffnlp/twitter-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = []
        for text in tqdm(X):
            encoded_input = self.tokenizer(text, return_tensors="pt")
            features = self.model(**encoded_input)
            features = features[0].detach().cpu().numpy()
            vectors.append(np.mean(features[0], axis=0))
        return np.vstack(vectors)
