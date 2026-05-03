import streamlit as st
import pandas as pd
from ui.config import THRESHOLDS, STRINGS

class CacheManager:
    """Isolates DataFrame generations under caching decorators mapping to step signatures"""
    
    @staticmethod
    def get_state_frame(state: dict, step: int):
        columns = ["Zone", "Car Slots", "Bike Slots", "Capacity", "Occupied", "Free", "Entries", "Exits", "Utilisation %", "Recommendation"]
        if not isinstance(state, dict) or not state:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(state).T.reset_index(names="Zone")
        if "capacity" not in frame and "total_slots" in frame:
            frame["capacity"] = frame["total_slots"]
        elif "capacity" in frame and "total_slots" in frame:
            capacity_values = pd.to_numeric(frame["capacity"], errors="coerce").fillna(0).astype(int)
            total_values = pd.to_numeric(frame["total_slots"], errors="coerce").fillna(0).astype(int)
            frame["capacity"] = capacity_values.where(capacity_values > 0, total_values)
        frame = frame.rename(
            columns={
                "capacity": "Capacity",
                "car_slots": "Car Slots",
                "bike_slots": "Bike Slots",
                "occupied": "Occupied",
                "free_slots": "Free",
                "entry": "Entries",
                "exit": "Exits",
            }
        )
        for column in ["Capacity", "Car Slots", "Bike Slots", "Occupied", "Entries", "Exits"]:
            if column not in frame:
                frame[column] = 0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
        frame["Car Slots"] = frame["Car Slots"].clip(lower=0)
        frame["Bike Slots"] = frame["Bike Slots"].clip(lower=0)
        slot_total = frame["Car Slots"] + frame["Bike Slots"]
        frame["Capacity"] = frame["Capacity"].where(frame["Capacity"] > 0, slot_total)
        frame["Capacity"] = frame["Capacity"].clip(lower=0)
        missing_split = (slot_total == 0) & (frame["Capacity"] > 0)
        frame.loc[missing_split, "Car Slots"] = frame.loc[missing_split, "Capacity"]
        split_total = frame["Car Slots"] + frame["Bike Slots"]
        mismatched_split = (frame["Capacity"] > 0) & (split_total != frame["Capacity"])
        frame.loc[mismatched_split, "Bike Slots"] = (
            frame.loc[mismatched_split, "Capacity"] - frame.loc[mismatched_split, "Car Slots"]
        ).clip(lower=0)
        split_total = frame["Car Slots"] + frame["Bike Slots"]
        still_mismatched = (frame["Capacity"] > 0) & (split_total != frame["Capacity"])
        frame.loc[still_mismatched, "Car Slots"] = frame.loc[still_mismatched, "Capacity"]
        frame.loc[still_mismatched, "Bike Slots"] = 0
        frame["Occupied"] = frame.apply(lambda row: max(0, min(int(row["Occupied"]), int(row["Capacity"]))), axis=1)
        frame["Free"] = frame["Capacity"] - frame["Occupied"]
        frame["Entries"] = frame["Entries"].clip(lower=0)
        frame["Exits"] = frame["Exits"].clip(lower=0)
        frame["Utilisation %"] = 0.0
        has_capacity = frame["Capacity"] > 0
        frame.loc[has_capacity, "Utilisation %"] = (
            (frame.loc[has_capacity, "Occupied"] / frame.loc[has_capacity, "Capacity"]) * 100
        ).round(1)
        
        def assign_recommendation(free):
            p = THRESHOLDS["free_slots"]["preferred"]
            o = THRESHOLDS["free_slots"]["overflow"]
            if free >= p:
                return STRINGS["recommendations"]["preferred"]
            elif free >= o:
                return STRINGS["recommendations"]["overflow"]
            return STRINGS["recommendations"]["avoid"]
            
        frame["Recommendation"] = frame["Free"].apply(assign_recommendation)
        return frame[columns]

    @staticmethod
    def get_transition_frame(zone_rows: list, step: int):
        if not zone_rows:
            return pd.DataFrame(columns=["Zone", "Before", "After", "Entries", "Exits", "Net Change"])
        frame = pd.DataFrame(zone_rows).rename(
            columns={
                "zone": "Zone",
                "occupied_before": "Before",
                "occupied_after": "After",
                "entry": "Entries",
                "exit": "Exits",
                "occupancy_change": "Net Change",
            }
        )
        for column in ["Zone", "Before", "After", "Entries", "Exits", "Net Change"]:
            if column not in frame:
                frame[column] = "" if column == "Zone" else 0
        return frame[["Zone", "Before", "After", "Entries", "Exits", "Net Change"]]

    @staticmethod
    def get_agent_frame(agent_interactions: list, step: int):
        rows = []
        for item in agent_interactions or []:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload", {})
            if isinstance(payload, dict):
                preview = ", ".join(f"{key}: {value}" for key, value in list(payload.items())[:3])
            else:
                preview = str(payload)
            why = item.get("Why", item.get("why", ""))
            if isinstance(why, dict):
                why = " | ".join(
                    f"{key}: {value}"
                    for key, value in list(why.items())[:3]
                    if value not in (None, "", [], {})
                )
            rows.append(
                {
                    "Agent": item.get("Agent", item.get("agent")),
                    "Mode": item.get("Mode", item.get("mode", "local")),
                    "Action Taken": item.get("Action Taken", item.get("message", "")),
                    "Why": why,
                    "Key Output": item.get("Key Output", preview[:130]),
                }
            )
        frame = pd.DataFrame(rows)
        for column in ["Agent", "Mode", "Action Taken", "Why", "Key Output"]:
            if column in frame:
                frame[column] = frame[column].astype(str)
        return frame

    @staticmethod
    def get_cycle_frame(recent_cycles: list, step: int):
        if not recent_cycles:
            return pd.DataFrame(columns=["Step", "Event", "Action", "Reward", "LLM Used", "LLM Status", "LLM Influence", "LLM Detail"])
        return pd.DataFrame(
            [
                {
                    "Step": cycle.get("step"),
                    "Event": cycle.get("event_context", {}).get("name", ""),
                    "Action": CacheManager._format_cycle_action(cycle),
                    "Reward": CacheManager._format_cycle_reward(cycle),
                    "LLM Used": CacheManager._format_cycle_llm_used(cycle),
                    "LLM Status": CacheManager._format_cycle_llm(cycle),
                    "LLM Influence": CacheManager._format_cycle_llm_influence(cycle),
                    "LLM Detail": CacheManager._format_cycle_reason(cycle),
                }
                for cycle in recent_cycles
            ]
        )

    @staticmethod
    def _format_cycle_reward(cycle: dict):
        reward = cycle.get("reward", {})
        value = None
        if isinstance(reward, dict):
            for key in ("agentic_reward_score", "reward_score", "environment_reward"):
                if reward.get(key) is not None:
                    value = reward.get(key)
                    break
        if value is None:
            value = cycle.get("reward_score", 0)
        try:
            return f"{float(value):+.2f}"
        except (TypeError, ValueError):
            return "+0.00"

    @staticmethod
    def _format_cycle_action(cycle: dict):
        action = cycle.get("execution_output", {}).get("final_action", {}) or cycle.get("action", {})
        kind = str(action.get("action", "none")).upper()
        if kind == "REDIRECT":
            return f"{action.get('from', '-')} -> {action.get('to', '-')} ({action.get('vehicles', 0)})"
        return kind

    @staticmethod
    def _format_cycle_llm(cycle: dict):
        planner = cycle.get("planner_output", {})
        source = planner.get("llm_source", "deterministic")
        if source == "gemini":
            return "Live Gemini"
        if source == "gemini_failed_fallback":
            return "Gemini Attempted"
        if planner.get("llm_advisory_used") and source == "gemini":
            return "Live Gemini"
        if planner.get("llm_requested"):
            return "Gemini Attempted"
        if source == "cached":
            return "Cached Gemini"
        if source in {"local_simulated", "simulated_edge_intelligence", "demo_simulated"}:
            return "Demo Gemini" if source == "demo_simulated" else "Local Reasoning"
        return "Local Reasoning"

    @staticmethod
    def _format_cycle_llm_used(cycle: dict):
        planner = cycle.get("planner_output", {})
        critic = cycle.get("critic_output", {})
        source = planner.get("llm_source", "deterministic")
        if source == "gemini":
            return "Yes"
        if source == "cached":
            return "Cached"
        if source == "gemini_failed_fallback":
            return "Attempted"
        if planner.get("llm_advisory_used") or critic.get("llm_advisory_used"):
            return "Yes"
        if planner.get("llm_requested") or critic.get("llm_requested"):
            return "Attempted"
        if planner.get("llm_source") == "demo_simulated":
            return "Demo"
        return "No"

    @staticmethod
    def _format_cycle_llm_influence(cycle: dict):
        planner = cycle.get("planner_output", {})
        source = planner.get("llm_source", "deterministic")
        if planner.get("llm_influence"):
            return "Modified"
        if source == "gemini":
            return "Confirmed"
        if planner.get("llm_advisory_used") and source == "gemini":
            return "Confirmed"
        if planner.get("llm_requested") and planner.get("llm_fallback_used"):
            return "Fallback"
        if source == "cached":
            return "Cached"
        if source == "demo_simulated":
            return "Demo advisory"
        if source in {"local_simulated", "simulated_edge_intelligence", "demo_simulated"}:
            return "Local Reasoning"
        return "-"

    @staticmethod
    def _format_cycle_reason(cycle: dict):
        planner = cycle.get("planner_output", {})
        action = cycle.get("execution_output", {}).get("final_action", {}) or cycle.get("action", {})
        text = (
            planner.get("llm_summary")
            or planner.get("rationale")
            or action.get("reason")
            or planner.get("llm_fallback_reason")
            or "Stable flow; local agents handled this step."
        )
        return str(text)[:260]

    @staticmethod
    def get_benchmark_frame(benchmark_data: dict, toggle: bool):
        # toggle used simply to force recache if someone clicks benchmark switch
        rows = []
        for item in (benchmark_data or {}).get("scenarios", []):
            rows.append(
                {
                    "Scenario": item.get("scenario"),
                    "Agentic Search Time": item.get("agentic", {}).get("avg_search_time_min", 0),
                    "Baseline Search Time": item.get("baseline", {}).get("avg_search_time_min", 0),
                    "Search Time Gain": item.get("delta_search_time", 0),
                    "Agentic Resilience": item.get("agentic", {}).get("avg_resilience_score", 0),
                    "Baseline Resilience": item.get("baseline", {}).get("avg_resilience_score", 0),
                    "Resilience Gain": item.get("delta_resilience", 0),
                    "Hotspot Reduction": item.get("delta_hotspots", 0),
                }
            )
        return pd.DataFrame(rows)

state_manager = CacheManager()
