from llm_reasoning import ask_llm_for_structured_json


class PlannerAgent:
    def plan(self, state, demand, insight, memory_metrics, tools):
        tool_calls = []
        goal_status = tools["get_goal_status"]()
        tool_calls.append("get_goal_status")
        pressure_report = tools["build_zone_pressure_report"](state, demand)
        tool_calls.append("build_zone_pressure_report")
        recent_cycles = tools["get_recent_cycles"]()
        tool_calls.append("get_recent_cycles")
        event_context = tools["get_event_context"]()
        tool_calls.append("get_event_context")
        scenario_mode = tools["get_scenario_mode"]()
        tool_calls.append("get_scenario_mode")
        best_zone = tools["suggest_best_zone"](state)
        tool_calls.append("suggest_best_zone")

        most_crowded = min(state, key=lambda zone: state[zone]["free_slots"])
        congested_zones = [zone for zone, data in state.items() if data["free_slots"] < 10]
        strategy = event_context.get("allocation_strategy", "Balanced utilisation")
        focus_zone = event_context.get("focus_zone", most_crowded)
        recommended_zone = event_context.get("recommended_zone", best_zone)
        source_zone = most_crowded
        destination_zone = recommended_zone if recommended_zone != source_zone else best_zone
        if destination_zone not in state or destination_zone == source_zone:
            destination_zone = best_zone
        learning_profile = tools["get_learning_profile"](
            scenario_mode=scenario_mode,
            from_zone=source_zone,
            to_zone=destination_zone,
        )
        tool_calls.append("get_learning_profile")
        safe_transfer_capacity = tools["estimate_transfer_capacity"](source_zone, destination_zone, 14)
        tool_calls.append("estimate_transfer_capacity")

        fallback_goal = goal_status or {
            "objective": "Reduce parking search time and keep congested zones to one or fewer over the next 5 steps.",
            "target_congested_zones": 1,
            "horizon_steps": 5,
            "priority_zone": source_zone,
            "target_search_time_min": 4.0,
        }

        scenario_bias = learning_profile.get("scenario_profile", {}).get("preferred_transfer_bias", 1.0)
        route_bias = learning_profile.get("route_profile", {}).get("success_bias", 1.0)
        global_bias = learning_profile.get("global_transfer_bias", 1.0)
        learned_bias = max(0.5, min(1.5, (scenario_bias + route_bias + global_bias) / 3))
        base_requested = max(
            0,
            min(14, max(0, 14 - state[source_zone]["free_slots"]) + int(demand.get(source_zone, 0) / 7)),
        )
        requested_vehicles = max(0, min(safe_transfer_capacity, int(round(base_requested * learned_bias))))
        fallback_plan = {
            "goal": fallback_goal,
            "analysis": {
                "most_crowded": most_crowded,
                "best_zone": best_zone,
                "congested_zones": congested_zones,
                "recent_cycles": len(recent_cycles),
                "event_name": event_context.get("name"),
                "safe_transfer_capacity": safe_transfer_capacity,
                "learned_bias": round(learned_bias, 2),
            },
            "strategy": strategy,
            "tool_calls": tool_calls,
            "tool_observations": {
                "goal_status": goal_status,
                "pressure_report": pressure_report,
                "event_context": event_context,
                "learning_profile": learning_profile,
                "safe_transfer_capacity": safe_transfer_capacity,
            },
            "proposed_action": {
                "action": "redirect" if requested_vehicles > 0 and source_zone != destination_zone else "none",
                "from": source_zone,
                "to": destination_zone,
                "vehicles": requested_vehicles,
                "reason": (
                    f"{strategy} recommends shifting incoming demand away from {source_zone} toward {destination_zone} during {event_context.get('name')}."
                ),
                "confidence": round(max(0.65, insight.get("confidence", 0.7)), 2),
            },
            "rationale": (
                f"Planner sees {len(congested_zones)} congested zones, the live hotspot is {source_zone}, "
                f"the event focus is {focus_zone}, and the learned transfer bias is {round(learned_bias, 2)}."
            ),
        }

        context = {
            "state": state,
            "demand": demand,
            "insight": insight,
            "memory_metrics": memory_metrics,
            "goal_status": goal_status,
            "recent_cycles": recent_cycles,
            "pressure_report": pressure_report,
            "event_context": event_context,
            "learning_profile": learning_profile,
            "safe_transfer_capacity": safe_transfer_capacity,
            "scenario_mode": scenario_mode,
        }
        schema_text = """
{
  "goal": {
    "objective": "string",
    "target_congested_zones": 1,
    "horizon_steps": 5,
    "priority_zone": "string",
    "target_search_time_min": 4.0
  },
  "analysis": {
    "most_crowded": "string",
    "best_zone": "string",
    "congested_zones": ["zone"]
  },
  "strategy": "string",
  "tool_calls": ["tool_name"],
  "tool_observations": {
    "goal_status": {},
    "pressure_report": {},
    "event_context": {},
    "learning_profile": {},
    "safe_transfer_capacity": 0
  },
  "proposed_action": {
    "action": "redirect or none",
    "from": "zone",
    "to": "zone",
    "vehicles": 0,
    "reason": "string",
    "confidence": 0.0
  },
  "rationale": "string"
}
"""
        llm_plan = ask_llm_for_structured_json(
            "PlannerAgent",
            context,
            schema_text,
            fallback_plan,
            system_instruction=(
                "You are the default planning brain. Use the supplied tools and observations first, "
                "respect safe_transfer_capacity, and only redirect when it improves the active goal."
            ),
        )
        llm_plan.setdefault("tool_calls", tool_calls)
        llm_plan.setdefault("tool_observations", fallback_plan["tool_observations"])
        return llm_plan
