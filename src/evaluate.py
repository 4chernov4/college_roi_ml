from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

from config import MODEL_PATH
from train_model import train_model

def evaluate_model():

    model_data = load(MODEL_PATH) if MODEL_PATH else None

    if model_data is None:
        model, X_test, y_test = train_model()
    else:
        model = model_data["model"]
        preprocessor = model_data["preprocessor"]

        from data_loader import load_data
        from preprocessing import preprocess_data

        df = load_data()
        _, X_test, _, y_test, _ = preprocess_data(df)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Качество модели:")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")

    return r2, mae, rmse
