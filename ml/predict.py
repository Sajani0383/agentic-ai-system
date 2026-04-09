import os
import warnings
from functools import lru_cache

import joblib
import pandas as pd

try:
    from sklearn.base import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = Warning

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "models", "demand_model.pkl")


@lru_cache(maxsize=1)
def _load_model():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            return joblib.load(model_path)
    except Exception:
        return None


def _fallback_predict_demand(hour, day, zone_id, vehicle_type=0):
    peak_hours = {8, 9, 10, 12, 13, 14, 17, 18}
    hour_factor = 28 if hour in peak_hours else 16
    weekday_factor = 6 if day in {1, 2, 3, 4, 5} else -4
    zone_factor = zone_id * 3
    vehicle_factor = vehicle_type * 2
    return max(8, hour_factor + weekday_factor + zone_factor + vehicle_factor)


def predict_demand(hour, day, zone_id, vehicle_type=0):
    model = _load_model()
    if model is None:
        return _fallback_predict_demand(hour, day, zone_id, vehicle_type)

    data = pd.DataFrame(
        [
            {
                "hour": hour,
                "day": day,
                "zone_id": zone_id,
                "vehicle_type": vehicle_type,
            }
        ]
    )

    try:
        prediction = model.predict(data)[0]
        return int(prediction)
    except Exception:
        return _fallback_predict_demand(hour, day, zone_id, vehicle_type)
