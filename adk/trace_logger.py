import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from threading import RLock


LOGGER = logging.getLogger("smart_parking.trace_logger")
LOGGER.addHandler(logging.NullHandler())


class TraceLogger:
    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, max_traces=500, storage_path=None, enable_persistence=True):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.max_traces = max(1, int(max_traces))
        self.storage_path = storage_path or os.path.join(base_dir, "memory", "trace_log.json")
        self.enable_persistence = enable_persistence
        self.traces = []
        self._lock = RLock()
        self._load()

    def log(self, step, event, data=None, level="INFO"):
        normalized_level = self._normalize_level(level)
        record = {
            "time": datetime.utcnow().isoformat(),
            "level": normalized_level,
            "step": step,
            "event": str(event),
            "data": deepcopy(data) if data is not None else {},
        }

        with self._lock:
            self.traces.append(record)
            self.traces = self.traces[-self.max_traces :]
            self._save()

        LOGGER.log(
            self.LEVELS[normalized_level],
            "step=%s event=%s data=%s",
            step,
            event,
            data,
        )
        return deepcopy(record)

    def debug(self, step, event, data=None):
        return self.log(step, event, data, level="DEBUG")

    def info(self, step, event, data=None):
        return self.log(step, event, data, level="INFO")

    def warning(self, step, event, data=None):
        return self.log(step, event, data, level="WARNING")

    def error(self, step, event, data=None):
        return self.log(step, event, data, level="ERROR")

    def get_traces(self, step=None, event=None, level=None, limit=None):
        with self._lock:
            traces = deepcopy(self.traces)

        if step is not None:
            traces = [trace for trace in traces if trace.get("step") == step]
        if event is not None:
            traces = [trace for trace in traces if trace.get("event") == event]
        if level is not None:
            normalized_level = self._normalize_level(level)
            traces = [trace for trace in traces if trace.get("level") == normalized_level]
        if limit is not None:
            traces = traces[-max(0, int(limit)) :]
        return traces

    def get_by_step(self, step, limit=None):
        return self.get_traces(step=step, limit=limit)

    def get_by_event(self, event, limit=None):
        return self.get_traces(event=event, limit=limit)

    def summary(self):
        with self._lock:
            traces = deepcopy(self.traces)

        counts_by_level = {}
        counts_by_event = {}
        for trace in traces:
            level = trace.get("level", "INFO")
            event = trace.get("event", "unknown")
            counts_by_level[level] = counts_by_level.get(level, 0) + 1
            counts_by_event[event] = counts_by_event.get(event, 0) + 1

        return {
            "total_traces": len(traces),
            "max_traces": self.max_traces,
            "oldest_time": traces[0]["time"] if traces else None,
            "latest_time": traces[-1]["time"] if traces else None,
            "counts_by_level": counts_by_level,
            "counts_by_event": counts_by_event,
            "persistence_enabled": self.enable_persistence,
            "storage_path": self.storage_path if self.enable_persistence else None,
        }

    def format_trace(self, trace):
        return (
            f"[{trace.get('time', '-')}] "
            f"{trace.get('level', 'INFO')} "
            f"step={trace.get('step', '-')} "
            f"event={trace.get('event', '-')} "
            f"data={trace.get('data', {})}"
        )

    def pretty(self, step=None, event=None, level=None, limit=20):
        return [
            self.format_trace(trace)
            for trace in self.get_traces(step=step, event=event, level=level, limit=limit)
        ]

    def clear(self):
        with self._lock:
            self.traces = []
            self._save()

    def _normalize_level(self, level):
        normalized = str(level or "INFO").upper()
        if normalized not in self.LEVELS:
            return "INFO"
        return normalized

    def _load(self):
        if not self.enable_persistence or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return

        traces = payload.get("traces", []) if isinstance(payload, dict) else []
        if isinstance(traces, list):
            self.traces = traces[-self.max_traces :]

    def _save(self):
        if not self.enable_persistence:
            return
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        payload = {
            "max_traces": self.max_traces,
            "saved_at": datetime.utcnow().isoformat(),
            "traces": self.traces,
        }
        temp_path = f"{self.storage_path}.tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        os.replace(temp_path, self.storage_path)


trace_logger = TraceLogger()
