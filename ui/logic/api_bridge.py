import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import streamlit as st


class BackendBridge:
    """HTTP bridge so Streamlit and the frontend share the same backend API state."""

    def __init__(self):
        self.base_url = os.getenv("PARKING_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        self.api_key = os.getenv("PARKING_API_KEY", "")
        self._cached_snapshot = {}

    def _build_url(self, path):
        separator = "&" if "?" in path else "?"
        return f"{self.base_url}{path}{separator}_ts={int(time.time() * 1000)}"

    def _request(self, method, path, payload=None):
        body = None
        headers = {
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._build_url(path), data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def get_snapshot(self):
        try:
            snapshot = self._request("GET", "/state")
            self._cached_snapshot = snapshot if isinstance(snapshot, dict) else {}
            return self._cached_snapshot
        except Exception as exc:
            logging.error("Backend Bridge crashed accessing /state: %s", exc)
            if self._cached_snapshot:
                st.warning("API connection dropped. Showing the last synchronized backend snapshot.")
                return self._cached_snapshot
            st.error("Lost connection to the backend API. Dashboard is in safe fallback mode.")
            return {}

    def _post(self, path, payload=None, error_message="Request failed"):
        try:
            result = self._request("POST", path, payload)
            if isinstance(result, dict):
                self._cached_snapshot = result.get("result", result) if path in {"/reset", "/scenario"} else self._cached_snapshot
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            st.error(f"{error_message}: {detail or exc.reason}")
            return None
        except Exception as exc:
            st.error(f"{error_message}: {exc}")
            return None

    def set_scenario(self, mode: str):
        result = self._post("/scenario", {"scenario_mode": mode}, "Failed to change scenario")
        return result is not None

    def set_llm_mode(self, mode: str):
        result = self._post("/llm/mode", {"llm_mode": mode}, "Failed to change LLM mode")
        return result is not None

    def set_force_llm(self, enabled: bool):
        result = self._post("/llm/force", {"enabled": enabled}, "Failed to toggle Strategic Overdrive")
        return result is not None

    def reset_llm_runtime_state(self):
        result = self._post("/llm/reset", error_message="Failed to reset AI quota state")
        return result is not None

    def step_simulation(self):
        return self._post("/step", error_message="Simulation tick failed") is not None

    def reset(self, clear_memory=False):
        result = self._post("/reset", {"clear_memory": clear_memory}, "Failed to reset environment")
        return result is not None

    def run_benchmark(self, episodes=3, steps=10):
        return self._post("/benchmark", {"episodes": episodes, "steps_per_episode": steps}, "Benchmark run failed") is not None

    def get_run_report(self):
        try:
            return self._request("GET", "/export/report")
        except Exception as exc:
            st.error(f"Report export failed: {exc}")
            return {}

    def ask(self, query: str):
        result = self._post("/run", {"input": query}, "Chat request failed")
        if isinstance(result, dict):
            if result.get("type") == "chat":
                return {"answer": result.get("message", "No answer returned.")}
            return {"answer": result.get("message", "Command completed.")}
        return {"answer": "API Error: request failed"}


api_bridge = BackendBridge()
