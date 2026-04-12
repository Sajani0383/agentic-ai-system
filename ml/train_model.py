import json
import os
import math
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
MODEL_VERSION = datetime.utcnow().strftime("demand_model_%Y%m%d_%H%M%S")

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "dataset", "parking_dataset.csv")
model_dir = os.path.join(base_dir, "..", "models")
model_path = os.path.join(model_dir, "demand_model.pkl")
metrics_path = os.path.join(model_dir, "demand_model_metrics.json")
versioned_model_path = os.path.join(model_dir, f"{MODEL_VERSION}.pkl")

os.makedirs(model_dir, exist_ok=True)


def _load_dataset():
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time", "occupied_slots", "zone", "vehicle_type"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["net_flow"] = df["entry_count"].fillna(0) - df["exit_count"].fillna(0)
    df["occupancy_ratio"] = (
        df["occupied_slots"].fillna(0) / df["total_slots"].replace(0, pd.NA)
    ).fillna(0)
    df["violation_flag"] = (df["violation_reported"].fillna("No") == "Yes").astype(int)
    return df


def _build_pipeline():
    numeric_features = [
        "hour",
        "day_of_week",
        "is_weekend",
        "total_slots",
        "avg_parking_duration_minutes",
        "entry_count",
        "exit_count",
        "parking_fee_collected",
        "occupancy_rate_percent",
        "net_flow",
        "occupancy_ratio",
        "violation_flag",
    ]
    categorical_features = ["zone", "vehicle_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    candidate_models = [
        RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        RandomForestRegressor(
            n_estimators=220,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ]

    return preprocessor, candidate_models, numeric_features + categorical_features


def _evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(math.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }


def train_model():
    os.makedirs(model_dir, exist_ok=True)
    df = _load_dataset()
    preprocessor, candidate_models, feature_columns = _build_pipeline()
    X = df[feature_columns]
    y = df["occupied_slots"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    best_pipeline = None
    best_metrics = None
    for candidate in candidate_models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", candidate),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = _evaluate_model(pipeline, X_test, y_test)
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_pipeline = pipeline
            best_metrics = metrics

    joblib.dump(best_pipeline, model_path)
    joblib.dump(best_pipeline, versioned_model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_columns,
        "target": "occupied_slots",
        "metrics": best_metrics,
        "saved_model_path": model_path,
        "versioned_model_path": versioned_model_path,
        "trained_at": datetime.utcnow().isoformat(),
        "model_family": type(best_pipeline.named_steps["model"]).__name__,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Model trained successfully")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train_model()
