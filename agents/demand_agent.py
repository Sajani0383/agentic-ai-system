from copy import deepcopy


class DemandAgent:
    DEFAULT_CONFIG = {
        "capacity_pressure_threshold": 0.72,
        "high_occupancy_threshold": 0.88,
        "flow_weight": 0.32,
        "scarcity_weight": 0.34,
        "trend_weight": 0.18,
        "event_weight": 0.16,
        "history_window": 6,
    }

    EVENT_SEVERITY_MULTIPLIERS = {
        "low": 1.0,
        "medium": 1.12,
        "high": 1.28,
        "critical": 1.42,
        "adaptive": 1.08,
    }

    PEAK_HOUR_MULTIPLIERS = {
        "morning_arrival": 1.22,
        "lunch_shift": 1.10,
        "evening_exit": 1.18,
        "normal": 1.0,
    }

    def __init__(self, config=None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.zone_history = {}
        self.feedback_bias = {}
        self.last_report = {}

    def predict(
        self,
        state,
        event_context=None,
        operational_signals=None,
        simulated_hour=None,
        historical_states=None,
        return_details=False,
    ):
        event_context = event_context or {}
        operational_signals = operational_signals or {}
        if historical_states:
            self._ingest_history(historical_states)
        self._record_state(state)

        raw_scores = {}
        zone_details = {}
        for zone, data in state.items():
            components = self._score_components(
                zone,
                data,
                event_context,
                operational_signals,
                simulated_hour,
            )
            raw_score = self._weighted_score(components)
            raw_scores[zone] = raw_score
            zone_details[zone] = components

        normalized_scores = self._normalize_scores(raw_scores)
        demand = {
            zone: int(round(normalized_scores[zone]))
            for zone in state
        }
        uncertainty = {
            zone: self._uncertainty(zone, zone_details[zone])
            for zone in state
        }
        confidence = {
            zone: round(1.0 - uncertainty[zone], 3)
            for zone in state
        }

        self.last_report = {
            "model_type": "adaptive rule-based demand forecaster",
            "parameters": deepcopy(self.config),
            "time_context": {
                "simulated_hour": simulated_hour,
                "time_bucket": self._time_bucket(simulated_hour),
                "time_multiplier": self._time_multiplier(simulated_hour),
            },
            "event_context": {
                "name": event_context.get("name"),
                "severity": event_context.get("severity"),
                "focus_zone": event_context.get("focus_zone"),
                "recommended_zone": event_context.get("recommended_zone"),
            },
            "raw_scores": {zone: round(score, 2) for zone, score in raw_scores.items()},
            "normalized_demand": demand,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "zone_details": zone_details,
            "history_lengths": {
                zone: len(history)
                for zone, history in self.zone_history.items()
            },
            "explanation": (
                "Demand combines live flow pressure, capacity scarcity, recent occupancy trend, "
                "event severity, time-of-day multiplier, dynamic signals, and feedback bias."
            ),
        }
        return deepcopy(self.last_report) if return_details else demand

    def update_from_feedback(self, predicted_demand, kpis=None):
        kpis = kpis or {}
        search_time = kpis.get("estimated_search_time_min", 0.0)
        queue_length = kpis.get("queue_length", 0)
        congestion_hotspots = kpis.get("congestion_hotspots", 0)
        pressure_signal = min(1.0, search_time / 6.0 + queue_length / 12.0 + congestion_hotspots / 10.0)

        for zone, demand_value in (predicted_demand or {}).items():
            current_bias = self.feedback_bias.get(zone, 1.0)
            direction = 0.04 if demand_value < 35 and pressure_signal > 0.6 else -0.02 if pressure_signal < 0.25 else 0.0
            self.feedback_bias[zone] = round(min(1.35, max(0.75, current_bias + direction)), 3)

    def get_last_report(self):
        return deepcopy(self.last_report)

    def _score_components(self, zone, data, event_context, operational_signals, simulated_hour):
        occupancy_ratio = self._occupancy_ratio(data)
        flow_pressure = max(0, data.get("entry", 0) - data.get("exit", 0)) / max(1, data.get("total_slots", 1))
        scarcity_pressure = self._scarcity_pressure(occupancy_ratio)
        trend_pressure = self._trend_pressure(zone)
        event_multiplier = self._event_multiplier(zone, event_context)
        time_multiplier = self._time_multiplier(simulated_hour)
        signal_multiplier = self._signal_multiplier(zone, operational_signals, event_context)
        feedback_bias = self.feedback_bias.get(zone, 1.0)

        return {
            "occupancy_ratio": round(occupancy_ratio, 4),
            "flow_pressure": round(min(1.0, flow_pressure), 4),
            "scarcity_pressure": round(scarcity_pressure, 4),
            "trend_pressure": round(trend_pressure, 4),
            "event_multiplier": round(event_multiplier, 4),
            "time_multiplier": round(time_multiplier, 4),
            "signal_multiplier": round(signal_multiplier, 4),
            "feedback_bias": round(feedback_bias, 4),
        }

    def _weighted_score(self, components):
        base_pressure = (
            components["flow_pressure"] * self.config["flow_weight"]
            + components["scarcity_pressure"] * self.config["scarcity_weight"]
            + components["trend_pressure"] * self.config["trend_weight"]
            + (components["event_multiplier"] - 1.0) * self.config["event_weight"]
        )
        adjusted = (
            base_pressure
            * components["event_multiplier"]
            * components["time_multiplier"]
            * components["signal_multiplier"]
            * components["feedback_bias"]
        )
        return max(0.0, min(100.0, adjusted * 100))

    def _normalize_scores(self, raw_scores):
        if not raw_scores:
            return {}
        max_score = max(raw_scores.values())
        if max_score <= 0:
            return {zone: 0 for zone in raw_scores}
        return {
            zone: max(0, min(100, (score / max_score) * 100))
            for zone, score in raw_scores.items()
        }

    def _scarcity_pressure(self, occupancy_ratio):
        threshold = self.config["capacity_pressure_threshold"]
        if occupancy_ratio <= threshold:
            return max(0.0, occupancy_ratio / max(1.0, threshold) * 0.35)
        high_threshold = self.config["high_occupancy_threshold"]
        pressure = 0.35 + (occupancy_ratio - threshold) / max(0.01, high_threshold - threshold) * 0.45
        if occupancy_ratio >= high_threshold:
            pressure += (occupancy_ratio - high_threshold) * 1.2
        return max(0.0, min(1.0, pressure))

    def _trend_pressure(self, zone):
        history = self.zone_history.get(zone, [])[-self.config["history_window"] :]
        if len(history) < 2:
            return 0.25
        first = history[0]["occupancy_ratio"]
        last = history[-1]["occupancy_ratio"]
        return max(0.0, min(1.0, 0.5 + (last - first) * 2.5))

    def _event_multiplier(self, zone, event_context):
        severity = event_context.get("severity", "low")
        multiplier = self.EVENT_SEVERITY_MULTIPLIERS.get(severity, 1.0)
        if zone == event_context.get("focus_zone"):
            multiplier += 0.16
        if zone == event_context.get("recommended_zone"):
            multiplier += 0.06
        zone_multipliers = event_context.get("zone_multipliers", {})
        multiplier *= zone_multipliers.get(zone, 1.0)
        return max(0.65, min(1.9, multiplier))

    def _signal_multiplier(self, zone, operational_signals, event_context):
        multiplier = 1.0
        if operational_signals.get("weather") == "Rain Surge":
            multiplier += 0.08
        if operational_signals.get("queue_length", 0) >= 4:
            multiplier += 0.10
        if operational_signals.get("blocked_zone") == zone:
            multiplier *= 0.35
        if operational_signals.get("vip_reserve_zone") == zone:
            multiplier *= 0.88
        if zone == event_context.get("recommended_zone") and operational_signals.get("queue_length", 0) >= 4:
            multiplier += 0.08
        return max(0.35, min(1.5, multiplier))

    def _time_multiplier(self, simulated_hour):
        bucket = self._time_bucket(simulated_hour)
        return self.PEAK_HOUR_MULTIPLIERS[bucket]

    def _time_bucket(self, simulated_hour):
        if simulated_hour in {8, 9, 10}:
            return "morning_arrival"
        if simulated_hour in {12, 13, 14}:
            return "lunch_shift"
        if simulated_hour in {17, 18, 19}:
            return "evening_exit"
        return "normal"

    def _uncertainty(self, zone, components):
        history_count = len(self.zone_history.get(zone, []))
        history_penalty = max(0.0, 0.35 - min(0.35, history_count * 0.06))
        volatility_penalty = min(0.25, abs(components["trend_pressure"] - 0.5) * 0.3)
        signal_penalty = 0.08 if components["signal_multiplier"] != 1.0 else 0.0
        return round(min(0.75, 0.18 + history_penalty + volatility_penalty + signal_penalty), 3)

    def _record_state(self, state):
        for zone, data in state.items():
            history = self.zone_history.setdefault(zone, [])
            history.append(
                {
                    "occupancy_ratio": self._occupancy_ratio(data),
                    "entry": data.get("entry", 0),
                    "exit": data.get("exit", 0),
                    "free_slots": data.get("free_slots", 0),
                }
            )
            self.zone_history[zone] = history[-self.config["history_window"] :]

    def _ingest_history(self, historical_states):
        for state in historical_states[-self.config["history_window"] :]:
            if isinstance(state, dict):
                self._record_state(state)

    def _occupancy_ratio(self, data):
        return data.get("occupied", 0) / max(1, data.get("total_slots", 1))
