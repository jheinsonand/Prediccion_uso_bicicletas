import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str = "total_users", lags=(1,2,3,24)):
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    df = df.dropna()
    return df