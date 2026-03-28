from pathlib import Path
import sys

# Ruta absoluta desde la ubicación de este archivo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import mlflow.sklearn
from train import train_model

def train_model_with_mlflow():
    # Tracking URI apuntando siempre a ROOT/mlruns
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("bike_sharing_forecast")

    with mlflow.start_run(run_name="random_forest_run"):
        model, rmse = train_model()

        # Loguear parámetros del modelo
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("cv_splits", 3)

        # Loguear métrica
        mlflow.log_metric("rmse", rmse)

        # Loguear el modelo
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Tags opcionales
        mlflow.set_tag("developer", "Jheinson_Marin")
        mlflow.set_tag("model_type", "random_forest")

        print(f"Run registrado en MLflow con RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_model_with_mlflow()