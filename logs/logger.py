import csv
import json
import os
from copy import deepcopy
from datetime import datetime


class SimulationLogger:
    DEFAULT_FIELDS = [
        "timestamp",
        "step",
        "log_type",
        "mode",
        "goal",
        "action",
        "policy_action",
        "planner_output",
        "critic_output",
        "execution_output",
        "environment_reward",
        "reward_score",
        "kpis",
        "notifications",
        "reasoning",
        "reasoning_source",
        "status",
        "error",
    ]

    def __init__(self, log_dir=None, file_name="simulation_logs.csv", batch_size=10, max_in_memory=500, max_file_rows=5000):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.log_dir = log_dir or os.path.join(base_dir, "logs", "simulation")
        self.log_file = os.path.join(self.log_dir, file_name)
        self.batch_size = max(1, int(batch_size))
        self.max_in_memory = max(10, int(max_in_memory))
        self.max_file_rows = max(1, int(max_file_rows))
        self.logs = []
        self.buffer = []
        self.schema = list(self.DEFAULT_FIELDS)
        self.last_error = ""
        os.makedirs(self.log_dir, exist_ok=True)

    def reset_logs(self):
        self.logs = []
        self.buffer = []
        self.last_error = ""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log_step(self, data, log_type="step"):
        record = self._build_record(data, log_type=log_type)
        self.logs.append(record)
        self.logs = self.logs[-self.max_in_memory :]
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()
        return deepcopy(record)

    def log_event(self, data, log_type="event"):
        return self.log_step(data, log_type=log_type)

    def flush(self):
        if not self.buffer:
            return True
        try:
            self._write_buffer()
            self.buffer = []
            self.last_error = ""
            return True
        except OSError as exc:
            self.last_error = str(exc)
            return False

    def get_logs(self, limit=None):
        items = deepcopy(self.logs)
        if limit is not None:
            items = items[-max(0, int(limit)) :]
        return items

    def get_status(self):
        return {
            "log_file": self.log_file,
            "buffered_records": len(self.buffer),
            "in_memory_records": len(self.logs),
            "last_error": self.last_error,
        }

    def _build_record(self, data, log_type):
        payload = deepcopy(data) if isinstance(data, dict) else {"payload": data}
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": payload.get("step_number", payload.get("step", "-")),
            "log_type": str(log_type or "step"),
            "mode": payload.get("mode", ""),
            "goal": self._serialize(payload.get("goal", {})),
            "action": self._serialize(payload.get("action", {})),
            "policy_action": self._serialize(payload.get("policy_action", {})),
            "planner_output": self._serialize(payload.get("planner_output", {})),
            "critic_output": self._serialize(payload.get("critic_output", {})),
            "execution_output": self._serialize(payload.get("execution_output", {})),
            "environment_reward": payload.get("environment_reward", 0),
            "reward_score": payload.get("reward_score", 0),
            "kpis": self._serialize(payload.get("kpis", {})),
            "notifications": self._serialize(payload.get("notifications", [])),
            "reasoning": payload.get("reasoning", ""),
            "reasoning_source": payload.get("reasoning_source", ""),
            "status": payload.get("status", "ok"),
            "error": payload.get("error", ""),
        }
        for key in record:
            if key not in self.schema:
                self.schema.append(key)
        return record

    def _write_buffer(self):
        os.makedirs(self.log_dir, exist_ok=True)
        existing_rows = self._read_existing_rows()
        merged_rows = existing_rows + self.buffer
        if len(merged_rows) > self.max_file_rows:
            merged_rows = merged_rows[-self.max_file_rows :]

        fields = list(self.schema)
        for row in merged_rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)

        with open(self.log_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in merged_rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    def _read_existing_rows(self):
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", newline="", encoding="utf-8") as file:
                return list(csv.DictReader(file))
        except OSError:
            return []

    def _serialize(self, value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        return value
