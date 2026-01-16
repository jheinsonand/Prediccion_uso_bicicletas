# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.database import read_data_sql
from src.features import add_time_features, add_lag_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_latest.pkl")

def get_dataset():
    query = """
    SELECT *
    FROM bike_sharing
    ORDER BY date_time
    """
    df = read_data_sql(query)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.set_index("date_time").sort_index()

    df = add_time_features(df)
    df = add_lag_features(df, target_col="total_users")

    target = "total_users"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target]
    return X, y

def train_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X, y = get_dataset()
    tscv = TimeSeriesSplit(n_splits=3)

    best_model = None
    best_rmse = np.inf

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"Fold RMSE: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    joblib.dump(best_model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH} con RMSE {best_rmse:.2f}")

    return best_model, best_rmse