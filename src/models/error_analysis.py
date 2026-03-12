import pandas as pd


def attach_predictions(df: pd.DataFrame, y_pred, label_col: str = "genre") -> pd.DataFrame:
    out = df.copy()
    out["predicted_genre"] = y_pred
    out["is_error"] = out[label_col] != out["predicted_genre"]
    return out


def error_rate_by_slice(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    return (
        df.groupby(slice_col, dropna=False)["is_error"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "error_rate", "count": "n"})
    )
