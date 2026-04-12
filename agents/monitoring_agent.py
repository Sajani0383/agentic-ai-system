from copy import deepcopy

from adk.trace_logger import trace_logger


class MonitoringAgent:
    REQUIRED_FIELDS = ("total_slots", "occupied", "entry", "exit")

    def __init__(self, logger=None):
        self.logger = logger or trace_logger
        self.last_observation = {}
        self.last_valid_state = {}

    def observe(self, source):
        step = getattr(source, "step_count", "-")
        source_type = type(source).__name__
        try:
            raw_state = self._extract_state(source)
            normalized_state = self._normalize_state(raw_state)
            observation = {
                "status": "ok",
                "source_type": source_type,
                "zones": len(normalized_state),
                "summary": self._build_summary(normalized_state),
            }
            self.last_valid_state = deepcopy(normalized_state)
            self.last_observation = observation
            self.logger.log(step, "monitoring_observation", observation, level="INFO")
            return normalized_state
        except Exception as exc:
            error_payload = {
                "status": "error",
                "source_type": source_type,
                "message": str(exc),
            }
            self.last_observation = error_payload
            self.logger.log(step, "monitoring_observation", error_payload, level="ERROR")
            if self.last_valid_state:
                return deepcopy(self.last_valid_state)
            return {"error": str(exc)}

    def get_last_observation(self):
        return deepcopy(self.last_observation)

    def _extract_state(self, source):
        if hasattr(source, "get_state"):
            return source.get_state()
        if isinstance(source, dict):
            return source
        raise TypeError("Unsupported source type for monitoring observation.")

    def _normalize_state(self, raw_state):
        if not isinstance(raw_state, dict) or not raw_state:
            raise ValueError("Observed state must be a non-empty dict of zones.")

        normalized = {}
        for zone, payload in raw_state.items():
            if not isinstance(zone, str) or not zone.strip():
                raise ValueError("Each observed zone must have a non-empty string name.")
            if not isinstance(payload, dict):
                raise ValueError(f"Zone '{zone}' must map to a dict payload.")

            missing_fields = [field for field in self.REQUIRED_FIELDS if field not in payload]
            if missing_fields:
                raise ValueError(f"Zone '{zone}' is missing required fields: {', '.join(missing_fields)}.")

            total_slots = self._to_int(payload["total_slots"], zone, "total_slots")
            occupied = self._to_int(payload["occupied"], zone, "occupied")
            entry = self._to_int(payload["entry"], zone, "entry")
            exit_count = self._to_int(payload["exit"], zone, "exit")
            free_slots = payload.get("free_slots", total_slots - occupied)
            free_slots = self._to_int(free_slots, zone, "free_slots")

            if total_slots < 0:
                raise ValueError(f"Zone '{zone}' has invalid total_slots: {total_slots}.")
            if occupied < 0 or occupied > total_slots:
                raise ValueError(f"Zone '{zone}' has invalid occupied count: {occupied}.")
            if free_slots < 0 or free_slots > total_slots:
                raise ValueError(f"Zone '{zone}' has invalid free_slots count: {free_slots}.")
            if occupied + free_slots != total_slots:
                free_slots = total_slots - occupied
            if entry < 0 or exit_count < 0:
                raise ValueError(f"Zone '{zone}' has invalid entry/exit counts.")

            normalized[zone] = {
                "total_slots": total_slots,
                "occupied": occupied,
                "free_slots": free_slots,
                "entry": entry,
                "exit": exit_count,
            }
        return normalized

    def _build_summary(self, state):
        best_zone = max(state, key=lambda zone: state[zone]["free_slots"])
        crowded_zone = min(state, key=lambda zone: state[zone]["free_slots"])
        return {
            "best_zone": best_zone,
            "most_crowded": crowded_zone,
            "total_free_slots": sum(zone["free_slots"] for zone in state.values()),
        }

    def _to_int(self, value, zone, field):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"Zone '{zone}' has non-numeric {field}: {value}.") from None
