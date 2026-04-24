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
        frame = frame.rename(
            columns={
                "total_slots": "Capacity",
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
        frame["Capacity"] = frame["Capacity"].clip(lower=1)
        frame["Car Slots"] = frame["Car Slots"].clip(lower=0)
        frame["Bike Slots"] = frame["Bike Slots"].clip(lower=0)
        frame["Occupied"] = frame.apply(lambda row: max(0, min(int(row["Occupied"]), int(row["Capacity"]))), axis=1)
        frame["Free"] = frame["Capacity"] - frame["Occupied"]
        frame["Entries"] = frame["Entries"].clip(lower=0)
        frame["Exits"] = frame["Exits"].clip(lower=0)
        frame["Utilisation %"] = ((frame["Occupied"] / frame["Capacity"]) * 100).round(1)
        
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
            return pd.DataFrame(columns=["Step", "Event", "Goal", "Planner", "Final", "Reasoning", "Reward", "LLM Used", "LLM Influence", "LLM Decision"])
        return pd.DataFrame(
            [
                {
                    "Step": cycle.get("step"),
                    "Event": cycle.get("event_context", {}).get("name", ""),
                    "Goal": cycle.get("goal", {}).get("objective", ""),
                    "Planner": cycle.get("planner_output", {}).get("proposed_action", {}).get("action", "none").upper(),
                    "Final": cycle.get("execution_output", {}).get("final_action", {}).get("action", "none").upper(),
                    "Reasoning": cycle.get("reasoning_budget", {}).get("budget_level", "local_only"),
                    "Reward": cycle.get("reward", {}).get("environment_reward", 0),
                    "LLM Used": "✅" if cycle.get("planner_output", {}).get("llm_advisory_used") else ("🔄" if cycle.get("planner_output", {}).get("llm_requested") else "❌"),
                    "LLM Influence": "🎯 Modified" if cycle.get("planner_output", {}).get("llm_influence") else "➖",
                    "LLM Decision": (
                        cycle.get("planner_output", {}).get("llm_summary")
                        or cycle.get("planner_output", {}).get("rationale")
                        or cycle.get("planner_output", {}).get("llm_fallback_reason")
                        or "Deterministic fallback applied."
                    )[:80],
                }
                for cycle in recent_cycles
            ]
        )

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
