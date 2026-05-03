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
        self._last_full_sync = 0.0

    def _build_url(self, path):
        separator = "&" if "?" in path else "?"
        return f"{self.base_url}{path}{separator}_ts={int(time.time() * 1000)}"

    def _request(self, method, path, payload=None, timeout=12):
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def clear_cache(self):
        self._cached_snapshot = {}
        self._last_full_sync = 0.0

    def get_snapshot(self):
        try:
            if self._cached_snapshot:
                live = self._request("GET", "/client-state", timeout=4)
                self._cached_snapshot = self._merge_live_snapshot(self._cached_snapshot, live)
                return self._cached_snapshot
            snapshot = self._request("GET", "/state", timeout=8)
            self._cached_snapshot = snapshot if isinstance(snapshot, dict) else {}
            self._last_full_sync = time.time()
            return self._cached_snapshot
        except Exception as exc:
            logging.error("Backend Bridge crashed accessing /state: %s", exc)
            if self._cached_snapshot:
                return self._cached_snapshot
            return {}

    def _merge_live_snapshot(self, base, live):
        if not isinstance(base, dict):
            base = {}
        if not isinstance(live, dict):
            return base
        merged = dict(base)
        current = live.get("current_state") if isinstance(live.get("current_state"), dict) else live
        live_keys = (
            "step",
            "updated_at",
            "blocks",
            "vehicles",
            "simulated_vehicles",
            "user_vehicles",
            "users",
            "events",
            "gates",
            "vehicle_stats",
            "actions",
            "movement_log",
            "alerts",
            "latest_decision",
            "decision_reason",
            "agent_thought",
            "learning",
            "llm",
            "agentic_integrity",
            "scenario_mode",
            "latest_result",
            "latest_transition",
            "metrics",
            "kpis",
            "event_context",
            "recent_cycles",
            "recent_states",
            "decision_explanation",
            "last_llm_decision",
            "llm_usage_summary",
            "reasoning_summary",
            "agent_loop_steps",
        )
        for key in live_keys:
            if key in current:
                merged[key] = current[key]
            elif key in live:
                merged[key] = live[key]
        merged["current_state"] = current
        blocks = current.get("blocks")
        latest_decision = current.get("latest_decision")
        if isinstance(blocks, dict):
            merged["state"] = blocks
            merged.setdefault("latest_result", {})
            if isinstance(merged["latest_result"], dict):
                merged["latest_result"]["state"] = blocks
        if isinstance(latest_decision, dict):
            merged.setdefault("latest_result", {})
            if isinstance(merged["latest_result"], dict):
                merged["latest_result"]["action"] = latest_decision
        if isinstance(live.get("latest_transition"), dict):
            merged["latest_transition"] = live["latest_transition"]
        else:
            merged.setdefault("latest_transition", {})
            if isinstance(merged["latest_transition"], dict):
                merged["latest_transition"]["step"] = current.get("step", merged["latest_transition"].get("step"))
                merged["latest_transition"]["timestamp"] = current.get("updated_at", merged["latest_transition"].get("timestamp"))
        return merged

    def _post(self, path, payload=None, error_message="Request failed", timeout=12):
        try:
            result = self._request("POST", path, payload, timeout=timeout)
            if isinstance(result, dict):
                self._cached_snapshot = result.get("result", result) if path in {"/reset", "/scenario", "/scenario/demo-pressure", "/scenario/force-full"} else self._cached_snapshot
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            st.error(f"{error_message}: {detail or exc.reason}")
            return None
        except Exception as exc:
            if "timed out" in str(exc).lower():
                st.warning(f"{error_message}: backend step is still running; showing the last synchronized state.")
            else:
                st.error(f"{error_message}: {exc}")
            return None

    def set_scenario(self, mode: str):
        result = self._post("/scenario", {"scenario_mode": mode}, "Failed to change scenario")
        return result is not None

    def force_full_scenario(self):
        result = self._post("/scenario/force-full", {}, "Failed to force full scenario")
        return result is not None

    def apply_demo_pressure(self, profile: str):
        result = self._post("/scenario/demo-pressure", {"profile": profile}, "Failed to apply demo pressure")
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

    def start_autonomy(self, interval_seconds=2.0):
        result = self._post(
            "/autonomy/start",
            {"interval_seconds": interval_seconds},
            "Failed to start backend autonomy",
        )
        if result is None:
            return False
        return True

    def stop_autonomy(self):
        result = self._post("/autonomy/stop", error_message="Failed to stop backend autonomy")
        return result is not None

    def get_autonomy_status(self):
        try:
            return self._request("GET", "/autonomy/status", timeout=4)
        except Exception:
            return {}

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
        result = self._post("/run", {"input": query}, "Chat request failed", timeout=45)
        if isinstance(result, dict):
            if result.get("type") == "chat":
                answer = result.get("answer") or result.get("message") or "No answer returned."
                return {
                    "answer": answer,
                    "source": result.get("source", "runtime_chat"),
                    "llm_used": bool(result.get("llm_used")),
                    "reason": result.get("reason", ""),
                }
            return {
                "answer": result.get("message", result.get("answer", "Command completed.")),
                "source": result.get("source", "runtime_command"),
                "llm_used": bool(result.get("llm_used")),
                "reason": result.get("reason", ""),
            }
        return {
            "answer": "Chat backend did not respond in time. The live simulation is still running, so the dashboard kept the last synchronized state.",
            "source": "dashboard_timeout",
            "llm_used": False,
            "reason": "The request timed out before a chat response reached Streamlit.",
        }


api_bridge = BackendBridge()
