import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import joblib
import pandas as pd
from database import read_data_sql
from features import add_time_features, add_lag_features

BASE_DIR = Path(__file__).resolve().parents[1]  # sube hasta la raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "model_latest.pkl"
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "predictions.csv"

def generate_forecast(horizon_hours=168):
    # Cargar últimos días de datos para crear lags
    query = """
    SELECT *
    FROM bike_sharing
    ORDER BY date_time
    """
    df = read_data_sql(query)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.set_index("date_time").sort_index()

    df = add_time_features(df)
    df = add_lag_features(df, target_col="users")

    model = joblib.load(MODEL_PATH)

    # Para simplificar: predecimos sobre el último día disponible
    X = df.iloc[-horizon_hours:][[c for c in df.columns if c != "users"]]
    y_hat = model.predict(X)

    pred_df = pd.DataFrame({
        "date_time": X.index,
        "forecast_users": y_hat
    })

    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predicciones guardadas en {PREDICTIONS_PATH}")

    return pred_df

if __name__ == "__main__":
    generate_forecast()