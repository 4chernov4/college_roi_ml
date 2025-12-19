from sklearn.ensemble import RandomForestRegressor
from joblib import dump

from config import MODEL_PATH, RF_PARAMS
from data_loader import load_data
from preprocessing import preprocess_data

def train_model():
    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = RandomForestRegressor(**RF_PARAMS)

    model.fit(X_train, y_train)

    dump(
        {
            "model": model,
            "preprocessor": preprocessor
        },
        MODEL_PATH
    )

    return model, X_test, y_test
