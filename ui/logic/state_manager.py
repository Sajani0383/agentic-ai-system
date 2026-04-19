import streamlit as st
import pandas as pd
from ui.config import THRESHOLDS, STRINGS

class CacheManager:
    """Isolates DataFrame generations under caching decorators mapping to step signatures"""
    
    @staticmethod
    @st.cache_data
    def get_state_frame(state: dict, step: int):
        frame = pd.DataFrame(state).T.reset_index(names="Zone")
        frame = frame.rename(
            columns={
                "total_slots": "Capacity",
                "occupied": "Occupied",
                "free_slots": "Free",
                "entry": "Entries",
                "exit": "Exits",
            }
        )
        # Avoid division by zero
        frame["Capacity"] = frame["Capacity"].replace(0, 1)
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
        return frame[["Zone", "Capacity", "Occupied", "Free", "Entries", "Exits", "Utilisation %", "Recommendation"]]

    @staticmethod
    @st.cache_data
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
        return frame[["Zone", "Before", "After", "Entries", "Exits", "Net Change"]]

    @staticmethod
    @st.cache_data
    def get_agent_frame(agent_interactions: list, step: int):
        rows = []
        for item in agent_interactions:
            payload = item.get("payload", {})
            if isinstance(payload, dict):
                preview = ", ".join(f"{key}: {value}" for key, value in list(payload.items())[:3])
            else:
                preview = str(payload)
            rows.append(
                {
                    "Agent": item.get("Agent", item.get("agent")),
                    "Mode": item.get("Mode", item.get("mode", "local")),
                    "Action Taken": item.get("Action Taken", item.get("message", "")),
                    "Why": item.get("Why", item.get("why", "")),
                    "Key Output": item.get("Key Output", preview[:130]),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    @st.cache_data
    def get_cycle_frame(recent_cycles: list, step: int):
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
                    "LLM Decision": cycle.get("planner_output", {}).get("llm_summary", "Deterministic fallback applied.")[:80],
                }
                for cycle in recent_cycles
            ]
        )

    @staticmethod
    @st.cache_data
    def get_benchmark_frame(benchmark_data: dict, toggle: bool):
        # toggle used simply to force recache if someone clicks benchmark switch
        rows = []
        for item in benchmark_data.get("scenarios", []):
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
