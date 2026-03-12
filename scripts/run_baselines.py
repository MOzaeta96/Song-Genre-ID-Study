from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.text_preprocessing import clean_lyrics
from src.models.evaluate import summarize_predictions
from src.models.train_baselines import get_baseline_specs, build_pipeline


def main():
    data_path = Path("data/processed/modeling_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    df = pd.read_csv(data_path)
    df["clean_lyrics"] = df["lyrics"].fillna("").map(clean_lyrics)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_lyrics"], df["genre"], test_size=0.2, random_state=42, stratify=df["genre"]
    )

    for spec in get_baseline_specs():
        pipe = build_pipeline(spec.estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = summarize_predictions(y_test, preds)
        print(spec.name, metrics)


if __name__ == "__main__":
    main()
