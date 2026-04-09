import json
import os
from copy import deepcopy
from datetime import datetime


class AgentMemory:
    def __init__(self, storage_path=None, max_history=200, max_cycles=200):
        base_dir = os.path.dirname(__file__)
        self.storage_path = storage_path or os.path.join(base_dir, "memory", "agent_memory_store.json")
        self.max_history = max_history
        self.max_cycles = max_cycles
        self.history = []
        self.cycles = []
        self.goal_history = []
        self.active_goal = {}
        self.learning_state = {
            "global_transfer_bias": 1.0,
            "scenario_profiles": {},
            "route_profiles": {},
            "recent_rewards": [],
            "q_table": [],
        }
        self._load()

    def add(self, state, transition=None, summary=None, step=None, kpis=None, notifications=None, event_context=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step if step is not None else len(self.history) + 1,
            "state": deepcopy(state),
            "summary": deepcopy(summary) if summary else {},
            "transition": deepcopy(transition) if transition else {},
            "kpis": deepcopy(kpis) if kpis else {},
            "notifications": deepcopy(notifications) if notifications else [],
            "event_context": deepcopy(event_context) if event_context else {},
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
        self._save()

    def log_cycle(self, cycle_data):
        self.cycles.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                **deepcopy(cycle_data),
            }
        )
        if len(self.cycles) > self.max_cycles:
            self.cycles = self.cycles[-self.max_cycles :]
        self._save()

    def set_goal(self, goal):
        comparable_current = {
            key: value for key, value in self.active_goal.items() if key != "timestamp"
        }
        if comparable_current == goal:
            return

        goal_record = {
            "timestamp": datetime.utcnow().isoformat(),
            **deepcopy(goal),
        }
        self.active_goal = goal_record
        self.goal_history.append(goal_record)
        self.goal_history = self.goal_history[-50:]
        self._save()

    def get_active_goal(self):
        return deepcopy(self.active_goal)

    def get_recent_states(self, limit=5):
        return deepcopy(self.history[-limit:])

    def get_recent_cycles(self, limit=5):
        return deepcopy(self.cycles[-limit:])

    def update_learning_signal(self, scenario_mode, action, reward_score, kpis=None):
        self.learning_state.setdefault("recent_rewards", []).append(round(reward_score, 2))
        self.learning_state["recent_rewards"] = self.learning_state["recent_rewards"][-50:]

        reward_window = self.learning_state["recent_rewards"]
        if reward_window:
            avg_reward = sum(reward_window) / len(reward_window)
            current_bias = float(self.learning_state.get("global_transfer_bias", 1.0))
            adjusted_bias = current_bias + (0.05 if avg_reward > 0 else -0.05)
            self.learning_state["global_transfer_bias"] = round(min(1.4, max(0.6, adjusted_bias)), 2)

        scenario_profiles = self.learning_state.setdefault("scenario_profiles", {})
        scenario_key = scenario_mode or "Unknown"
        scenario_profile = scenario_profiles.setdefault(
            scenario_key,
            {
                "steps": 0,
                "avg_reward": 0.0,
                "avg_search_time": 0.0,
                "avg_allocation_success": 0.0,
                "preferred_transfer_bias": 1.0,
            },
        )
        scenario_profile["steps"] += 1
        steps = scenario_profile["steps"]
        scenario_profile["avg_reward"] = round(
            ((scenario_profile["avg_reward"] * (steps - 1)) + reward_score) / steps,
            2,
        )
        if kpis:
            scenario_profile["avg_search_time"] = round(
                ((scenario_profile["avg_search_time"] * (steps - 1)) + kpis.get("estimated_search_time_min", 0.0)) / steps,
                2,
            )
            scenario_profile["avg_allocation_success"] = round(
                (
                    (scenario_profile["avg_allocation_success"] * (steps - 1))
                    + kpis.get("allocation_success_pct", 0.0)
                )
                / steps,
                2,
            )
        scenario_bias = scenario_profile.get("preferred_transfer_bias", 1.0)
        scenario_profile["preferred_transfer_bias"] = round(
            min(1.5, max(0.5, scenario_bias + (0.08 if reward_score > 0 else -0.08))),
            2,
        )

        if action and action.get("action") == "redirect":
            route_profiles = self.learning_state.setdefault("route_profiles", {})
            route_key = f"{action.get('from')}->{action.get('to')}"
            route_profile = route_profiles.setdefault(
                route_key,
                {
                    "attempts": 0,
                    "avg_reward": 0.0,
                    "success_bias": 1.0,
                },
            )
            route_profile["attempts"] += 1
            attempts = route_profile["attempts"]
            route_profile["avg_reward"] = round(
                ((route_profile["avg_reward"] * (attempts - 1)) + reward_score) / attempts,
                2,
            )
            route_profile["success_bias"] = round(
                min(1.6, max(0.4, route_profile["success_bias"] + (0.07 if reward_score > 0 else -0.07))),
                2,
            )

        self._save()

    def get_learning_profile(self, scenario_mode=None, from_zone=None, to_zone=None):
        scenario_profiles = self.learning_state.get("scenario_profiles", {})
        route_profiles = self.learning_state.get("route_profiles", {})
        route_key = None
        if from_zone and to_zone:
            route_key = f"{from_zone}->{to_zone}"
        return {
            "global_transfer_bias": round(float(self.learning_state.get("global_transfer_bias", 1.0)), 2),
            "recent_reward_avg": round(
                sum(self.learning_state.get("recent_rewards", []) or [0.0])
                / max(1, len(self.learning_state.get("recent_rewards", []))),
                2,
            ),
            "scenario_profile": deepcopy(scenario_profiles.get(scenario_mode, {})) if scenario_mode else {},
            "route_profile": deepcopy(route_profiles.get(route_key, {})) if route_key else {},
        }

    def set_q_table(self, q_table):
        self.learning_state["q_table"] = deepcopy(q_table)
        self._save()

    def get_q_table(self):
        return deepcopy(self.learning_state.get("q_table", []))

    def get_metrics(self):
        if not self.history:
            return {
                "steps": 0,
                "avg_free_slots": 0.0,
                "congestion_events": 0,
                "avg_space_utilisation_pct": 0.0,
                "avg_search_time_min": 0.0,
                "allocation_success_pct": 0.0,
                "avg_reward_score": 0.0,
                "active_goal": self.active_goal or {},
                "goal_updates": len(self.goal_history),
                "learning_profile": deepcopy(self.learning_state),
            }

        congestion_count = 0
        total_free = 0
        count = 0
        utilisation_total = 0.0
        search_time_total = 0.0
        allocation_total = 0.0
        kpi_count = 0
        reward_total = 0.0
        reward_count = 0

        for entry in self.history:
            state = entry["state"]
            for zone in state:
                free_slots = state[zone]["free_slots"]
                total_free += free_slots
                count += 1
                if free_slots < 10:
                    congestion_count += 1
            kpis = entry.get("kpis", {})
            if kpis:
                utilisation_total += kpis.get("space_utilisation_pct", 0.0)
                search_time_total += kpis.get("estimated_search_time_min", 0.0)
                allocation_total += kpis.get("allocation_success_pct", 0.0)
                kpi_count += 1
        for cycle in self.cycles:
            reward_total += cycle.get("reward", {}).get("reward_score", 0.0)
            reward_count += 1

        avg_free = total_free / count if count else 0
        return {
            "steps": len(self.history),
            "avg_free_slots": round(avg_free, 2),
            "congestion_events": congestion_count,
            "avg_space_utilisation_pct": round(utilisation_total / kpi_count, 2) if kpi_count else 0.0,
            "avg_search_time_min": round(search_time_total / kpi_count, 2) if kpi_count else 0.0,
            "allocation_success_pct": round(allocation_total / kpi_count, 2) if kpi_count else 0.0,
            "avg_reward_score": round(reward_total / reward_count, 2) if reward_count else 0.0,
            "active_goal": deepcopy(self.active_goal),
            "goal_updates": len(self.goal_history),
            "learning_profile": self.get_learning_profile(),
        }

    def export(self):
        return {
            "history": deepcopy(self.history),
            "cycles": deepcopy(self.cycles),
            "goal_history": deepcopy(self.goal_history),
            "active_goal": deepcopy(self.active_goal),
            "learning_state": deepcopy(self.learning_state),
        }

    def load_export(self, payload):
        self.history = deepcopy(payload.get("history", []))
        self.cycles = deepcopy(payload.get("cycles", []))
        self.goal_history = deepcopy(payload.get("goal_history", []))
        self.active_goal = deepcopy(payload.get("active_goal", {}))
        self.learning_state = deepcopy(payload.get("learning_state", self.learning_state))
        self._save()

    def reset(self, persist=True):
        self.history = []
        self.cycles = []
        self.goal_history = []
        self.active_goal = {}
        self.learning_state = {
            "global_transfer_bias": 1.0,
            "scenario_profiles": {},
            "route_profiles": {},
                "recent_rewards": [],
                "q_table": [],
        }
        if persist:
            self._save()

    def _load(self):
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return

        self.history = payload.get("history", [])
        self.cycles = payload.get("cycles", [])
        self.goal_history = payload.get("goal_history", [])
        self.active_goal = payload.get("active_goal", {})
        self.learning_state = payload.get("learning_state", self.learning_state)

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        payload = {
            "history": self.history,
            "cycles": self.cycles,
            "goal_history": self.goal_history,
            "active_goal": self.active_goal,
            "learning_state": self.learning_state,
        }
        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
