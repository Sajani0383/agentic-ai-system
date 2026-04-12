from copy import deepcopy

from llm_reasoning import ask_llm_for_structured_json


class PlannerAgent:
    DEFAULTS = {
        "base_horizon_steps": 3,
        "max_horizon_steps": 5,
        "base_congestion_threshold": 10,
        "base_transfer_cap": 14,
        "target_search_time_min": 4.0,
    }

    def plan(self, state, demand, insight, memory_metrics, tools):
        tool_calls = []
        tool_observations = {}

        goal_status = self._call_tool(tools, "get_goal_status", default={}, tool_calls=tool_calls)
        tool_observations["goal_status"] = goal_status

        event_context = self._call_tool(tools, "get_event_context", default={}, tool_calls=tool_calls)
        tool_observations["event_context"] = event_context

        scenario_mode = self._call_tool(tools, "get_scenario_mode", default="Unknown", tool_calls=tool_calls)
        operational_signals = self._call_tool(tools, "get_operational_signals", default={}, tool_calls=tool_calls)
        recent_cycles = self._call_tool(tools, "get_recent_cycles", default=[], tool_calls=tool_calls)
        best_zone = self._call_tool(
            tools,
            "suggest_best_zone",
            state,
            default=max(state, key=lambda zone: state[zone]["free_slots"]) if state else None,
            tool_calls=tool_calls,
        )
        pressure_report = self._call_tool(
            tools,
            "build_zone_pressure_report",
            state,
            demand,
            default=self._build_pressure_report(state, demand),
            tool_calls=tool_calls,
        )
        tool_observations["pressure_report"] = pressure_report

        analysis = self._build_analysis(state, demand, insight, event_context, operational_signals, recent_cycles, best_zone)
        source_zone = analysis["most_crowded"]
        destination_zone = analysis["recommended_destination"]

        learning_profile = self._call_tool(
            tools,
            "get_learning_profile",
            default={},
            keyword_args={
                "scenario_mode": scenario_mode,
                "from_zone": source_zone,
                "to_zone": destination_zone,
            },
            tool_calls=tool_calls,
        )
        tool_observations["learning_profile"] = learning_profile

        adaptive_limits = self._build_adaptive_limits(state, demand, event_context, operational_signals, insight, learning_profile)
        safe_transfer_capacity = self._call_tool(
            tools,
            "estimate_transfer_capacity",
            source_zone,
            destination_zone,
            adaptive_limits["max_transfer"],
            default=min(
                adaptive_limits["max_transfer"],
                state.get(source_zone, {}).get("entry", 0),
                state.get(destination_zone, {}).get("free_slots", 0),
            ) if source_zone in state and destination_zone in state else 0,
            tool_calls=tool_calls,
        )
        tool_observations["safe_transfer_capacity"] = safe_transfer_capacity
        tool_observations["operational_signals"] = operational_signals

        goal = self._build_goal(goal_status, source_zone, analysis, adaptive_limits)
        temporal_reasoning = self._build_temporal_reasoning(
            state,
            demand,
            insight,
            event_context,
            operational_signals,
            recent_cycles,
            adaptive_limits,
        )
        primary_action = self._build_primary_action(
            state,
            demand,
            insight,
            event_context,
            source_zone,
            destination_zone,
            learning_profile,
            safe_transfer_capacity,
            adaptive_limits,
            temporal_reasoning,
        )
        alternative_actions = self._build_alternatives(
            state,
            demand,
            source_zone,
            destination_zone,
            event_context,
            operational_signals,
            tools,
            adaptive_limits,
            tool_calls,
        )
        scoring = self._score_plan(primary_action, analysis, temporal_reasoning, insight, safe_transfer_capacity, adaptive_limits)
        action_sequence = self._build_action_sequence(primary_action, alternative_actions, goal, temporal_reasoning)
        planner_feedback = self._build_learning_feedback(primary_action, scoring, temporal_reasoning, learning_profile)

        fallback_plan = {
            "goal": goal,
            "analysis": analysis,
            "strategy": event_context.get("allocation_strategy", "Balanced utilisation"),
            "tool_calls": tool_calls,
            "tool_observations": tool_observations,
            "adaptive_limits": adaptive_limits,
            "uncertainty_assessment": {
                "planner_confidence": scoring["confidence"],
                "risk_probability": scoring["risk_probability"],
                "entropy": round(float(insight.get("uncertainty", {}).get("entropy", 0.0)), 3),
                "confidence_gap": insight.get("uncertainty", {}).get("confidence_gap", 0.0),
            },
            "temporal_reasoning": temporal_reasoning,
            "scoring": scoring,
            "proposed_action": primary_action,
            "alternative_actions": alternative_actions,
            "action_sequence": action_sequence,
            "planner_feedback": planner_feedback,
            "llm_advisory_used": False,
            "rationale": (
                f"Planner selected {source_zone} as the live hotspot, targeted {destination_zone} as the safest relief zone, "
                f"and projected pressure trend '{temporal_reasoning['pressure_trend']}' over {goal['horizon_steps']} steps."
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
            "operational_signals": operational_signals,
            "learning_profile": learning_profile,
            "adaptive_limits": adaptive_limits,
            "safe_transfer_capacity": safe_transfer_capacity,
            "temporal_reasoning": temporal_reasoning,
            "analysis": analysis,
            "scoring": scoring,
            "alternative_actions": alternative_actions,
            "scenario_mode": scenario_mode,
        }
        schema_text = """
{
  "strategy": "string",
  "proposed_action": {
    "action": "redirect or none",
    "from": "zone",
    "to": "zone",
    "vehicles": 0,
    "reason": "string",
    "confidence": 0.0
  },
  "alternative_actions": [
    {
      "action": "redirect or none",
      "from": "zone",
      "to": "zone",
      "vehicles": 0,
      "reason": "string"
    }
  ],
  "rationale": "string"
}
"""
        llm_advisory = ask_llm_for_structured_json(
            "PlannerAgent",
            context,
            schema_text,
            fallback_plan,
            system_instruction=(
                "You are an advisory planner. Use the deterministic analysis as the primary anchor. "
                "You may refine wording, propose a safer smaller redirect, or suggest better alternatives, "
                "but do not exceed safe_transfer_capacity or change the source/destination away from observed conditions without a strong reason."
            ),
        )
        final_plan = self._merge_llm_advisory(fallback_plan, llm_advisory, safe_transfer_capacity)
        final_plan.setdefault("tool_calls", tool_calls)
        final_plan.setdefault("tool_observations", tool_observations)
        final_plan.setdefault("adaptive_limits", adaptive_limits)
        final_plan.setdefault("alternative_actions", alternative_actions)
        final_plan.setdefault("action_sequence", action_sequence)
        final_plan.setdefault("scoring", scoring)
        final_plan.setdefault("temporal_reasoning", temporal_reasoning)
        final_plan.setdefault("planner_feedback", planner_feedback)
        return final_plan

    def _build_analysis(self, state, demand, insight, event_context, operational_signals, recent_cycles, best_zone):
        most_crowded = min(state, key=lambda zone: state[zone]["free_slots"])
        congestion_threshold = self._dynamic_congestion_threshold(state, operational_signals)
        congested_zones = [
            zone for zone, data in state.items() if data["free_slots"] <= congestion_threshold
        ]
        recommended_zone = event_context.get("recommended_zone", best_zone)
        destination = recommended_zone if recommended_zone in state and recommended_zone != most_crowded else best_zone
        if destination == most_crowded:
            destination = max(
                (zone for zone in state if zone != most_crowded),
                key=lambda zone: state[zone]["free_slots"],
                default=most_crowded,
            )
        return {
            "most_crowded": most_crowded,
            "best_zone": best_zone,
            "recommended_destination": destination,
            "congested_zones": congested_zones,
            "recent_cycles": len(recent_cycles or []),
            "event_name": event_context.get("name"),
            "event_focus_zone": event_context.get("focus_zone"),
            "queue_length": operational_signals.get("queue_length", 0),
            "blocked_zone": operational_signals.get("blocked_zone"),
            "bayesian_confidence": round(float(insight.get("confidence", 0.0)), 3),
        }

    def _build_goal(self, goal_status, source_zone, analysis, adaptive_limits):
        if goal_status:
            goal = deepcopy(goal_status)
            goal.setdefault("priority_zone", source_zone)
            goal.setdefault("target_search_time_min", self.DEFAULTS["target_search_time_min"])
            goal.setdefault("horizon_steps", adaptive_limits["horizon_steps"])
            return goal
        return {
            "objective": "Reduce parking search time and keep congested zones to one or fewer over the next few planning steps.",
            "target_congested_zones": 1,
            "horizon_steps": adaptive_limits["horizon_steps"],
            "priority_zone": source_zone,
            "target_search_time_min": self.DEFAULTS["target_search_time_min"],
            "active_hotspots": len(analysis["congested_zones"]),
        }

    def _build_adaptive_limits(self, state, demand, event_context, operational_signals, insight, learning_profile):
        source_zone = min(state, key=lambda zone: state[zone]["free_slots"])
        severity = event_context.get("severity", "low")
        severity_multiplier = {
            "low": 0.9,
            "medium": 1.0,
            "high": 1.15,
            "critical": 1.25,
            "adaptive": 1.05,
        }.get(severity, 1.0)
        queue_boost = 1 + min(0.2, operational_signals.get("queue_length", 0) * 0.03)
        learned_bias = max(
            0.65,
            min(
                1.4,
                (
                    learning_profile.get("global_transfer_bias", 1.0)
                    + learning_profile.get("scenario_profile", {}).get("preferred_transfer_bias", 1.0)
                    + learning_profile.get("route_profile", {}).get("success_bias", 1.0)
                )
                / 3,
            ),
        )
        base_transfer = self.DEFAULTS["base_transfer_cap"] + max(0, operational_signals.get("queue_length", 0) - 2)
        max_transfer = int(round(base_transfer * severity_multiplier * queue_boost * learned_bias))
        max_transfer = max(4, min(24, max_transfer))
        horizon_steps = min(
            self.DEFAULTS["max_horizon_steps"],
            self.DEFAULTS["base_horizon_steps"] + (1 if severity in {"high", "critical"} else 0) + (1 if operational_signals.get("queue_length", 0) >= 4 else 0),
        )
        congestion_threshold = self._dynamic_congestion_threshold(state, operational_signals)
        source_pressure = demand.get(source_zone, 0)
        return {
            "max_transfer": max_transfer,
            "congestion_threshold": congestion_threshold,
            "horizon_steps": horizon_steps,
            "learned_bias": round(learned_bias, 3),
            "source_pressure": source_pressure,
            "uncertainty_penalty": round(min(0.25, float(insight.get("uncertainty", {}).get("entropy", 0.0)) / 10), 3),
        }

    def _build_primary_action(
        self,
        state,
        demand,
        insight,
        event_context,
        source_zone,
        destination_zone,
        learning_profile,
        safe_transfer_capacity,
        adaptive_limits,
        temporal_reasoning,
    ):
        learned_bias = adaptive_limits["learned_bias"]
        destination_free = state.get(destination_zone, {}).get("free_slots", 0)
        source_free = state.get(source_zone, {}).get("free_slots", 0)
        base_requested = max(
            0,
            min(
                adaptive_limits["max_transfer"],
                max(0, adaptive_limits["congestion_threshold"] + 4 - source_free) + int(demand.get(source_zone, 0) / 6),
            ),
        )
        uncertainty_discount = max(0.6, 1.0 - adaptive_limits["uncertainty_penalty"])
        projected_escalation = 1.1 if temporal_reasoning["projected_queue_peak"] >= 4 else 1.0
        route_bias = learning_profile.get("route_profile", {}).get("success_bias", 1.0)
        requested = int(round(base_requested * learned_bias * route_bias * uncertainty_discount * projected_escalation))
        vehicles = max(0, min(safe_transfer_capacity, requested, destination_free))
        confidence = max(
            0.45,
            min(
                0.95,
                float(insight.get("confidence", 0.7)) * uncertainty_discount + (0.06 if vehicles > 0 else -0.08),
            ),
        )
        action_type = "redirect" if vehicles > 0 and source_zone != destination_zone else "none"
        return {
            "action": action_type,
            "from": source_zone,
            "to": destination_zone,
            "vehicles": vehicles,
            "reason": (
                f"{event_context.get('allocation_strategy', 'Balanced utilisation')} prioritizes relieving {source_zone} "
                f"toward {destination_zone} under projected {temporal_reasoning['pressure_trend']} pressure."
            ),
            "confidence": round(confidence, 3),
        }

    def _build_alternatives(
        self,
        state,
        demand,
        source_zone,
        destination_zone,
        event_context,
        operational_signals,
        tools,
        adaptive_limits,
        tool_calls,
    ):
        blocked_zone = operational_signals.get("blocked_zone")
        candidates = [
            zone for zone in state
            if zone != source_zone and zone != blocked_zone
        ]
        candidates.sort(
            key=lambda zone: (
                zone == event_context.get("recommended_zone"),
                state[zone]["free_slots"],
                -demand.get(zone, 0),
            ),
            reverse=True,
        )

        alternatives = []
        for zone in candidates[:3]:
            safe_capacity = self._call_tool(
                tools,
                "estimate_transfer_capacity",
                source_zone,
                zone,
                adaptive_limits["max_transfer"],
                default=min(
                    adaptive_limits["max_transfer"],
                    state[source_zone]["entry"],
                    state[zone]["free_slots"],
                ),
                tool_calls=tool_calls if zone != destination_zone else None,
            )
            vehicles = max(0, min(safe_capacity, max(1, int(demand.get(source_zone, 0) / 7))))
            alternatives.append(
                {
                    "action": "redirect" if vehicles > 0 else "none",
                    "from": source_zone,
                    "to": zone,
                    "vehicles": vehicles,
                    "reason": (
                        f"Fallback route to {zone} keeps backup capacity available"
                        if zone != destination_zone
                        else f"Primary route to {zone} remains the safest option"
                    ),
                }
            )
        if not alternatives:
            alternatives.append({"action": "none", "from": source_zone, "to": destination_zone, "vehicles": 0, "reason": "No safe backup route is available."})
        return alternatives

    def _score_plan(self, action, analysis, temporal_reasoning, insight, safe_transfer_capacity, adaptive_limits):
        vehicles = max(0, int(action.get("vehicles", 0) or 0))
        capacity_score = min(1.0, vehicles / max(1, safe_transfer_capacity)) if safe_transfer_capacity else 0.0
        hotspot_relief = min(
            1.0,
            max(0.0, (adaptive_limits["congestion_threshold"] + 4 - analysis["queue_length"]) / max(1, adaptive_limits["congestion_threshold"] + 4)),
        )
        projected_pressure = temporal_reasoning["projected_queue_peak"] / 6
        uncertainty = min(1.0, float(insight.get("uncertainty", {}).get("entropy", 0.0)) / 3)
        benefit_score = round(
            min(1.0, 0.45 * capacity_score + 0.35 * hotspot_relief + 0.20 * min(1.0, projected_pressure + 0.2)),
            3,
        )
        risk_probability = round(
            min(0.99, 0.20 + uncertainty * 0.35 + max(0, analysis["queue_length"] - 2) * 0.06),
            3,
        )
        efficiency_score = round(max(0.0, min(1.0, benefit_score - risk_probability * 0.35 + 0.15)), 3)
        confidence = round(max(0.35, min(0.95, action.get("confidence", 0.65) - risk_probability * 0.12)), 3)
        return {
            "benefit_score": benefit_score,
            "risk_probability": risk_probability,
            "efficiency_score": efficiency_score,
            "confidence": confidence,
            "safe_transfer_capacity": safe_transfer_capacity,
        }

    def _build_temporal_reasoning(
        self,
        state,
        demand,
        insight,
        event_context,
        operational_signals,
        recent_cycles,
        adaptive_limits,
    ):
        pressure_now = sum(1 for zone in state.values() if zone["free_slots"] <= adaptive_limits["congestion_threshold"])
        recent_queue = [
            cycle.get("kpis", {}).get("queue_length", 0)
            for cycle in (recent_cycles or [])
            if isinstance(cycle, dict)
        ]
        avg_recent_queue = round(sum(recent_queue) / max(1, len(recent_queue)), 2) if recent_queue else 0.0
        projected_queue_peak = max(
            operational_signals.get("queue_length", 0),
            int(round(avg_recent_queue + (1 if event_context.get("severity") in {"high", "critical"} else 0))),
        )
        source_zone = min(state, key=lambda zone: state[zone]["free_slots"])
        projected_hotspot_pressure = min(100, int(round(demand.get(source_zone, 0) * (1 + adaptive_limits["uncertainty_penalty"]))))
        if projected_queue_peak >= 5 or projected_hotspot_pressure >= 75:
            pressure_trend = "escalating"
        elif projected_queue_peak <= 2 and pressure_now <= 1:
            pressure_trend = "stable"
        else:
            pressure_trend = "elevated"
        return {
            "pressure_trend": pressure_trend,
            "projected_queue_peak": projected_queue_peak,
            "recent_average_queue": avg_recent_queue,
            "projected_hotspot_pressure": projected_hotspot_pressure,
            "forecast_horizon_steps": adaptive_limits["horizon_steps"],
            "time_factor": "event_peak" if event_context.get("severity") in {"high", "critical"} else "normal_window",
            "bayesian_entropy": round(float(insight.get("uncertainty", {}).get("entropy", 0.0)), 3),
        }

    def _build_action_sequence(self, primary_action, alternatives, goal, temporal_reasoning):
        sequence = [
            {
                "step": 1,
                "phase": "stabilize",
                "action": deepcopy(primary_action),
                "success_condition": f"Queue stays at or below {max(3, temporal_reasoning['projected_queue_peak'])}.",
            },
            {
                "step": 2,
                "phase": "monitor",
                "action": {"action": "observe"},
                "success_condition": f"Congested zones remain at or below {goal.get('target_congested_zones', 1)}.",
            },
        ]
        fallback_action = next((action for action in alternatives if action.get("action") == "redirect"), {"action": "none"})
        sequence.append(
            {
                "step": 3,
                "phase": "fallback",
                "action": fallback_action,
                "success_condition": "Use fallback route only if the primary route underperforms or queue pressure escalates.",
            }
        )
        return sequence

    def _build_learning_feedback(self, action, scoring, temporal_reasoning, learning_profile):
        return {
            "route": f"{action.get('from')}->{action.get('to')}" if action.get("from") and action.get("to") else None,
            "recommended_bias_adjustment": round(0.04 if scoring["benefit_score"] > scoring["risk_probability"] else -0.05, 3),
            "planner_confidence": scoring["confidence"],
            "risk_probability": scoring["risk_probability"],
            "temporal_trend": temporal_reasoning["pressure_trend"],
            "prior_route_profile": deepcopy(learning_profile.get("route_profile", {})),
        }

    def _merge_llm_advisory(self, fallback_plan, llm_advisory, safe_transfer_capacity):
        if not isinstance(llm_advisory, dict) or llm_advisory.get("goal"):
            return fallback_plan

        merged = deepcopy(fallback_plan)
        llm_action = llm_advisory.get("proposed_action", {})
        if self._is_safe_llm_action(merged["proposed_action"], llm_action, safe_transfer_capacity):
            merged["proposed_action"] = llm_action
            merged["llm_advisory_used"] = True
        if isinstance(llm_advisory.get("alternative_actions"), list) and llm_advisory["alternative_actions"]:
            merged["alternative_actions"] = llm_advisory["alternative_actions"][:3]
            merged["llm_advisory_used"] = True
        if isinstance(llm_advisory.get("strategy"), str) and llm_advisory["strategy"].strip():
            merged["strategy"] = llm_advisory["strategy"].strip()
        if isinstance(llm_advisory.get("rationale"), str) and llm_advisory["rationale"].strip():
            merged["rationale"] = llm_advisory["rationale"].strip()
        return merged

    def _is_safe_llm_action(self, fallback_action, llm_action, safe_transfer_capacity):
        if not isinstance(llm_action, dict):
            return False
        if fallback_action.get("action") == "none":
            return llm_action.get("action") == "none"
        if llm_action.get("action") == "none":
            return True
        return (
            llm_action.get("action") == "redirect"
            and llm_action.get("from") == fallback_action.get("from")
            and llm_action.get("to") == fallback_action.get("to")
            and 0 <= int(llm_action.get("vehicles", 0) or 0) <= int(min(safe_transfer_capacity, fallback_action.get("vehicles", 0) or 0))
        )

    def _dynamic_congestion_threshold(self, state, operational_signals):
        free_slots = [zone["free_slots"] for zone in state.values()]
        if not free_slots:
            return self.DEFAULTS["base_congestion_threshold"]
        average_free = sum(free_slots) / len(free_slots)
        queue_adjustment = 1 if operational_signals.get("queue_length", 0) >= 4 else 0
        return max(6, min(16, int(round((average_free * 0.3) + 4 + queue_adjustment))))

    def _build_pressure_report(self, state, demand):
        return {
            zone: {
                "free_slots": data["free_slots"],
                "occupied": data["occupied"],
                "demand_pressure": demand.get(zone, 0),
            }
            for zone, data in state.items()
        }

    def _call_tool(self, tools, name, *args, default=None, keyword_args=None, tool_calls=None):
        keyword_args = keyword_args or {}
        tool = tools.get(name) if isinstance(tools, dict) else None
        if tool_calls is not None:
            tool_calls.append({"tool": name, "used": callable(tool)})
        if not callable(tool):
            return deepcopy(default)
        try:
            result = tool(*args, **keyword_args) if keyword_args else tool(*args)
            return result if result is not None else deepcopy(default)
        except Exception:
            return deepcopy(default)
