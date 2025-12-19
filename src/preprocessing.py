import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    META_COLUMNS,
    PROCESSED_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

def preprocess_data(df):
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN] + META_COLUMNS]

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    numeric_features = FEATURE_COLUMNS

    categorical_features = []

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cleaned_df = df[FEATURE_COLUMNS + [TARGET_COLUMN] + META_COLUMNS]
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
