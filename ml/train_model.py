import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# FIXED PATHS
base_dir = os.path.dirname(__file__)

data_path = os.path.join(base_dir, "..", "dataset", "parking_dataset.csv")
model_dir = os.path.join(base_dir, "..", "models")
model_path = os.path.join(model_dir, "demand_model.pkl")

# CREATE models folder if not exists
os.makedirs(model_dir, exist_ok=True)

# LOAD DATA
df = pd.read_csv(data_path)

# FEATURE ENGINEERING
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day

# ENCODE
df["zone_id"] = df["zone"].astype("category").cat.codes
df["vehicle_type"] = df["vehicle_type"].astype("category").cat.codes

# FEATURES & TARGET
X = df[["hour", "day", "zone_id", "vehicle_type"]]
y = df["occupied_slots"]

# MODEL
model = RandomForestRegressor()
model.fit(X, y)

# SAVE MODEL
joblib.dump(model, model_path)

print("✅ Model trained and saved successfully!")