import joblib
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "models", "demand_model.pkl")

model = joblib.load(model_path)

def predict_demand(hour, day, zone_id, vehicle_type=0):
    data = pd.DataFrame([{
        "hour": hour,
        "day": day,
        "zone_id": zone_id,
        "vehicle_type": vehicle_type
    }])

    prediction = model.predict(data)[0]
    return int(prediction)