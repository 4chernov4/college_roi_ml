import pandas as pd
from joblib import load

from config import (
    MODEL_PATH,
    OUTPUT_RANKING_PATH,
    FEATURE_COLUMNS,
    META_COLUMNS
)
from data_loader import load_data


def calculate_roi():

    model_data = load(MODEL_PATH)
    model = model_data["model"]
    preprocessor = model_data["preprocessor"]
    
    df = load_data()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["COSTT4_A"])

    X = df[FEATURE_COLUMNS]
    X_processed = preprocessor.transform(X)

    df["predicted_income"] = model.predict(X_processed)

    df["ROI"] = df["predicted_income"] / df["COSTT4_A"]

    result_df = df[META_COLUMNS + [
        "COSTT4_A",
        "predicted_income",
        "ROI"
    ]]

    result_df = result_df.replace([float("inf"), -float("inf")], pd.NA)
    result_df = result_df.dropna()

    result_df = result_df.sort_values(by="ROI", ascending=False)

    result_df.to_csv(OUTPUT_RANKING_PATH, index=False)

    print("Рейтинг колледжей сохранён в:", OUTPUT_RANKING_PATH)

    return result_df
