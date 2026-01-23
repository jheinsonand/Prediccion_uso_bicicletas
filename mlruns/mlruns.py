

# import mlflow
# from train import train_model
# import mlflow.sklearn

# def train_model_with_mlflow():
#     mlflow.set_experiment("bike_sharing_forecast")

#     with mlflow.start_run():
#         model, rmse = train_model()
#         mlflow.log_metric("rmse", rmse)
#         mlflow.sklearn.log_model(model, "random_forest_model")

# if __name__ == "__main__":
#     train_model_with_mlflow()

import sys
from pathlib import Path

# mlruns.py está en: ROOT/mlruns/mlruns.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sube: mlruns -> ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))       # para importar train.py y demás desde src

import mlflow
import mlflow.sklearn

from train import train_model  # ahora se resuelve desde ROOT/src

def train_model_with_mlflow():
    mlruns_dir = PROJECT_ROOT / "mlruns"

    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("bike_sharing_forecast")

    with mlflow.start_run(run_name="random_forest_run"):
        model, rmse = train_model()
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.set_tag({"developer":"Jheinson_Marin", "model":"random_forest"})

if __name__ == "__main__":
    train_model_with_mlflow()