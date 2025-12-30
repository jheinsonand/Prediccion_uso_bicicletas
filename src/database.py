# src/database.py
import os
import sqlite3
import pandas as pd

GITHUB_CSV_URL = "https://raw.githubusercontent.com/jheinsonand/Prediccion_uso_bicicletas/refs/heads/main/data/raw/bike_sharing_dataset_clean.csv"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "bike_sharing.db")

def create_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def load_raw_from_github_to_db(table_name: str = "bike_sharing"):
    """Descarga el CSV desde GitHub y lo carga a una tabla SQLite."""
    print("Descargando datos desde GitHub...")
    df = pd.read_csv(GITHUB_CSV_URL)
    
    conn = create_connection()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Datos cargados en tabla '{table_name}'.")

def read_data_sql(query: str) -> pd.DataFrame:
    """Ejecuta una consulta SQL sencilla y devuelve un DataFrame."""
    conn = create_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df