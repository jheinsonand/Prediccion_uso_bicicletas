import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prefect import flow, task

from database import load_raw_from_github_to_db
from train import train_model

@task
def task_load_data():
    load_raw_from_github_to_db()

@task
def task_train_model():
    model, rmse = train_model()
    return rmse

@flow(name="bike_sharing_training_flow")
def training_flow():
    task_load_data()
    rmse = task_train_model()
    print(f"Flujo completado. RMSE: {rmse}")
    return rmse

if __name__ == "__main__":
    training_flow()