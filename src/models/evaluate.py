from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def summarize_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def class_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(report).T.reset_index(names="label")
