from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


@dataclass
class BaselineSpec:
    name: str
    estimator: object


def get_baseline_specs() -> list[BaselineSpec]:
    return [
        BaselineSpec("logistic_regression", LogisticRegression(max_iter=2000)),
        BaselineSpec("linear_svm", LinearSVC()),
        BaselineSpec("naive_bayes", MultinomialNB()),
    ]


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ("model", estimator),
    ])


def train_single_model(df: pd.DataFrame, text_col: str, label_col: str, estimator: object) -> Pipeline:
    pipe = build_pipeline(estimator)
    pipe.fit(df[text_col], df[label_col])
    return pipe
