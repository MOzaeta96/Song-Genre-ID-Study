import pandas as pd


def load_song_table(path: str) -> pd.DataFrame:
    """Load the base song table from CSV."""
    df = pd.read_csv(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=rename_map)
