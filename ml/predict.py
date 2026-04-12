import os
import json
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
metrics_path = os.path.join(base_dir, "..", "models", "demand_model_metrics.json")

FALLBACK_CONFIG = {
    "peak_hour_value": 28,
    "off_peak_value": 16,
    "weekday_bonus": 6,
    "weekend_penalty": -4,
    "zone_weight": 3,
    "vehicle_weight": 2,
    "min_prediction": 8,
}

PREDICTION_STATUS = {
    "model_loaded": False,
    "model_available": os.path.exists(model_path),
    "model_version": None,
    "last_error": "",
    "last_mode": "uninitialized",
}


def _set_prediction_status(**updates):
    PREDICTION_STATUS.update(updates)


@lru_cache(maxsize=1)
def _load_model():
    if not os.path.exists(model_path):
        _set_prediction_status(
            model_loaded=False,
            model_available=False,
            model_version="missing_model_file",
            last_error=f"Model file not found at {model_path}",
            last_mode="fallback",
        )
        return None


@lru_cache(maxsize=1)
def _load_model_metadata():
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(model_path)
        _set_prediction_status(
            model_loaded=True,
            model_available=True,
            model_version=getattr(model, "__class__", type(model)).__name__,
            last_error="",
            last_mode="model",
        )
        return model
    except Exception as exc:
        _set_prediction_status(
            model_loaded=False,
            model_available=False,
            model_version="load_failed",
            last_error=f"{type(exc).__name__}: {exc}",
            last_mode="fallback",
        )
        return None


def _validate_inputs(hour, day, zone_id, vehicle_type=0):
    if not isinstance(hour, int) or hour < 0 or hour > 23:
        raise ValueError("hour must be an integer between 0 and 23")
    if not isinstance(day, int) or day < 1 or day > 7:
        raise ValueError("day must be an integer between 1 and 7")
    if not isinstance(zone_id, int) or zone_id < 0:
        raise ValueError("zone_id must be a non-negative integer")
    if not isinstance(vehicle_type, int) or vehicle_type < 0:
        raise ValueError("vehicle_type must be a non-negative integer")


def _fallback_predict_demand(hour, day, zone_id, vehicle_type=0):
    peak_hours = {8, 9, 10, 12, 13, 14, 17, 18}
    hour_factor = FALLBACK_CONFIG["peak_hour_value"] if hour in peak_hours else FALLBACK_CONFIG["off_peak_value"]
    weekday_factor = FALLBACK_CONFIG["weekday_bonus"] if day in {1, 2, 3, 4, 5} else FALLBACK_CONFIG["weekend_penalty"]
    zone_factor = zone_id * FALLBACK_CONFIG["zone_weight"]
    vehicle_factor = vehicle_type * FALLBACK_CONFIG["vehicle_weight"]
    return max(FALLBACK_CONFIG["min_prediction"], hour_factor + weekday_factor + zone_factor + vehicle_factor)


def predict_demand_details(hour, day, zone_id, vehicle_type=0):
    _validate_inputs(hour, day, zone_id, vehicle_type)
    model = _load_model()
    metadata = _load_model_metadata()
    fallback_value = _fallback_predict_demand(hour, day, zone_id, vehicle_type)

    if model is None:
        return {
            "prediction": int(fallback_value),
            "confidence": 0.42,
            "mode": "fallback",
            "fallback_used": True,
            "model_version": metadata.get("model_version", PREDICTION_STATUS.get("model_version")),
            "last_error": PREDICTION_STATUS.get("last_error", ""),
            "input": {
                "hour": hour,
                "day": day,
                "zone_id": zone_id,
                "vehicle_type": vehicle_type,
            },
        }

    data = pd.DataFrame([_build_feature_row(hour, day, zone_id, vehicle_type, metadata)])

    try:
        prediction = float(model.predict(data)[0])
        confidence = _estimate_confidence(prediction, fallback_value)
        _set_prediction_status(last_error="", last_mode="model")
        return {
            "prediction": int(round(max(0, prediction))),
            "confidence": confidence,
            "mode": "model",
            "fallback_used": False,
            "model_version": metadata.get("model_version", PREDICTION_STATUS.get("model_version")),
            "last_error": "",
            "input": {
                "hour": hour,
                "day": day,
                "zone_id": zone_id,
                "vehicle_type": vehicle_type,
            },
        }
    except Exception as exc:
        _set_prediction_status(last_error=f"{type(exc).__name__}: {exc}", last_mode="fallback")
        return {
            "prediction": int(fallback_value),
            "confidence": 0.38,
            "mode": "fallback",
            "fallback_used": True,
            "model_version": metadata.get("model_version", PREDICTION_STATUS.get("model_version")),
            "last_error": PREDICTION_STATUS.get("last_error", ""),
            "input": {
                "hour": hour,
                "day": day,
                "zone_id": zone_id,
                "vehicle_type": vehicle_type,
            },
        }


def predict_demand(hour, day, zone_id, vehicle_type=0):
    return predict_demand_details(hour, day, zone_id, vehicle_type)["prediction"]


def predict_demand_batch(records):
    results = []
    for record in records or []:
        details = predict_demand_details(
            int(record.get("hour", 0)),
            int(record.get("day", 1)),
            int(record.get("zone_id", 0)),
            int(record.get("vehicle_type", 0)),
        )
        results.append(details)
    return results


def get_prediction_status():
    status = dict(PREDICTION_STATUS)
    metadata = _load_model_metadata()
    if metadata.get("model_version"):
        status["model_version"] = metadata["model_version"]
    return status


def _estimate_confidence(prediction, fallback_value):
    difference = abs(float(prediction) - float(fallback_value))
    confidence = max(0.45, min(0.95, 0.9 - difference / 80))
    return round(confidence, 3)


def _build_feature_row(hour, day, zone_id, vehicle_type, metadata):
    day_of_week = max(0, min(6, day - 1))
    feature_row = {
        "hour": hour,
        "day": day,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week in {5, 6} else 0,
        "zone_id": zone_id,
        "zone": f"Zone_{zone_id}",
        "vehicle_type": str(vehicle_type),
        "total_slots": 150,
        "avg_parking_duration_minutes": 90,
        "entry_count": fallback_value_proxy(hour, day, zone_id, vehicle_type),
        "exit_count": max(0, fallback_value_proxy(hour, day, zone_id, vehicle_type) - 3),
        "parking_fee_collected": 0.0,
        "occupancy_rate_percent": 50.0,
        "net_flow": 3,
        "occupancy_ratio": 0.5,
        "violation_flag": 0,
    }
    feature_columns = metadata.get("feature_columns", [])
    if feature_columns:
        return {column: feature_row.get(column) for column in feature_columns}
    return feature_row


def fallback_value_proxy(hour, day, zone_id, vehicle_type):
    return _fallback_predict_demand(hour, day, zone_id, vehicle_type)
