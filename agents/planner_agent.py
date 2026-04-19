from copy import deepcopy

from llm_reasoning import ask_llm_for_structured_json


class PlannerAgent:
    DEFAULTS = {
        "base_horizon_steps": 3,
        "max_horizon_steps": 5,
        "base_congestion_threshold": 12,
        "base_transfer_cap": 18,
        "target_search_time_min": 4.0,
    }

    def plan(self, state, demand, insight, memory_metrics, tools, reasoning_budget=None):
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

        # 1. Fetch Learning Profile EARLY (includes blocked_routes)
        learning_profile = self._call_tool(
            tools,
            "get_learning_profile",
            default={},
            keyword_args={
                "scenario_mode": scenario_mode,
            },
            tool_calls=tool_calls,
        )
        tool_observations["learning_profile"] = learning_profile
        blocked_routes = learning_profile.get("blocked_routes", [])
        consolidated_insights = learning_profile.get("consolidated_insights", [])

        # 2. Pass learning_profile to analysis for hard filtering
        analysis_pre = self._build_analysis(state, demand, insight, event_context, operational_signals, recent_cycles, best_zone, learning_profile)
        source_zone = analysis_pre["most_crowded"]
        destination_zone = analysis_pre["recommended_destination"]

        # Add recently failed zones to avoidance pool
        recently_failed_zones = []
        for f in learning_profile.get("recent_failures", []):
            if f.get("failed") and f.get("to"):
                recently_failed_zones.append(f["to"])
        recently_failed_zones = list(set(recently_failed_zones))

        # 3. Decision Filtering: If the primary route is blocked OR the destination is recently failed
        route_key = f"{source_zone}->{destination_zone}"
        learning_override = False
        if route_key in blocked_routes or destination_zone in recently_failed_zones:
            other_zones = [z for z in state if z != source_zone and z != destination_zone and z not in recently_failed_zones]
            if not other_zones:
                # If everything failed, at least avoid officially blocked ones
                other_zones = [z for z in state if z != source_zone and z != destination_zone and f"{source_zone}->{z}" not in blocked_routes]
            
            if other_zones:
                alt_destination = max(other_zones, key=lambda z: state[z]["free_slots"])
                if f"{source_zone}->{alt_destination}" not in blocked_routes:
                    destination_zone = alt_destination
                    analysis_pre["recommended_destination"] = destination_zone
                    analysis_pre["memory_avoidance_triggered"] = True
                    analysis_pre["avoided_route"] = route_key
                    learning_override = True

        # 4. Pattern-Based VETO: Check consolidated insights for explicit destination warnings
        for insight_text in consolidated_insights:
            if f"to {destination_zone}" in insight_text or destination_zone in insight_text:
                other_zones = [z for z in state if z != source_zone and z != destination_zone and z not in recently_failed_zones]
                if other_zones:
                    alt = max(other_zones, key=lambda z: state[z]["free_slots"])
                    destination_zone = alt
                    analysis_pre["recommended_destination"] = destination_zone
                    analysis_pre["memory_pattern_veto"] = True
                    analysis_pre["veto_reason"] = insight_text
                    learning_override = True
                    break

        analysis = analysis_pre
        analysis["learning_override_active"] = learning_override
        adaptive_limits = self._build_adaptive_limits(state, demand, event_context, operational_signals, insight, learning_profile)
        safe_transfer_capacity = self._call_tool(
            tools,
            "estimate_transfer_capacity",
            source_zone,
            destination_zone,
            adaptive_limits["max_transfer"],
            default=min(
                adaptive_limits["max_transfer"],
                max(1, int(demand.get(source_zone, 30) / 5)),
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
            learning_profile,
        )
        scoring = self._score_plan(primary_action, analysis, temporal_reasoning, insight, safe_transfer_capacity, adaptive_limits, learning_profile)
        action_sequence = self._build_action_sequence(primary_action, alternative_actions, goal, temporal_reasoning)
        planner_feedback = self._build_learning_feedback(primary_action, scoring, temporal_reasoning, learning_profile)

        dynamic_strategy = event_context.get("allocation_strategy", "Balanced utilisation")
        if learning_profile.get("recent_reward_avg", 0.0) < -0.3:
            dynamic_strategy = "Emergency Capacity Recovery"
        elif learning_profile.get("last_reward", 0.0) < -0.1:
            dynamic_strategy = "Corrective Risk Mitigation"

        fallback_plan = {
            "goal": goal,
            "analysis": analysis,
            "strategy": dynamic_strategy,
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
                f"I have identified {source_zone} as the primary network pressure point during the current {event_context.get('name', 'simulation')} window. "
                f"To prevent queue escalation, I am proactively routing incoming vehicles to {destination_zone}, which offers the highest safety buffer. "
                f"My projections suggest pressure will remain '{temporal_reasoning['pressure_trend']}' over the next {goal['horizon_steps']} steps, "
                f"making this redirect essential for maintaining campus mobility."
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
        llm_advisory = fallback_plan
        llm_gate = reasoning_budget or {}
        llm_strategy = llm_gate.get("planner_llm_strategy", "deterministic")
        llm_requested = bool(llm_gate.get("allow_planner_llm") and llm_strategy == "gemini")
        if llm_strategy == "gemini" and llm_requested:
            llm_advisory = ask_llm_for_structured_json(
                "PlannerAgent",
                context,
                schema_text,
                fallback_plan,
                system_instruction=(
                    "You are an advisory planner. Analyze the deterministic analysis. "
                    "If the current performance is suboptimal or risk is acceptable, you MUST OVERRIDE the deterministic planner "
                    "by proposing a completely different 'from', 'to', or 'vehicles' parameters to change the plan. "
                    "Do not exceed safe_transfer_capacity."
                ),
                force=llm_gate.get("force_llm", False),
            )
        elif llm_strategy == "cached":
            llm_advisory = llm_gate.get("cached_planner_advisory", fallback_plan)
        elif llm_strategy == "local_simulated":
            llm_advisory = llm_gate.get("local_simulated_advisory", fallback_plan)
        elif llm_strategy == "demo_simulated":
            llm_advisory = llm_gate.get("local_simulated_advisory", fallback_plan)

        final_plan = self._merge_llm_advisory(fallback_plan, llm_advisory, safe_transfer_capacity)
        final_plan.setdefault("tool_calls", tool_calls)
        final_plan.setdefault("tool_observations", tool_observations)
        final_plan.setdefault("adaptive_limits", adaptive_limits)
        final_plan.setdefault("alternative_actions", alternative_actions)
        final_plan.setdefault("action_sequence", action_sequence)
        final_plan.setdefault("scoring", scoring)
        final_plan.setdefault("temporal_reasoning", temporal_reasoning)
        final_plan.setdefault("planner_feedback", planner_feedback)
        final_plan["llm_requested"] = llm_requested
        final_plan["forced_live_attempt"] = bool(llm_gate.get("force_llm", False) and llm_strategy == "gemini")
        if llm_strategy == "cached":
            final_plan["decision_mode"] = "cached_llm_advisory"
            final_plan["llm_source"] = "cached"
            final_plan["llm_advisory_used"] = True
        elif llm_strategy == "local_simulated":
            final_plan["decision_mode"] = "autonomous_edge_optimization"
            final_plan["llm_source"] = "local_simulated"
            final_plan["llm_advisory_used"] = False
        elif llm_strategy == "demo_simulated":
            final_plan["decision_mode"] = "demo_simulated_gemini"
            final_plan["llm_source"] = "demo_simulated"
            final_plan["llm_advisory_used"] = True
        elif llm_strategy == "gemini" and llm_requested:
            final_plan["decision_mode"] = "autonomous_localized_fallback"
            final_plan["llm_source"] = "gemini_failed_fallback"
            final_plan["llm_advisory_used"] = False
        elif final_plan.get("llm_advisory_used"):
            final_plan["decision_mode"] = "distributed_cloud_briefing"
            final_plan["llm_source"] = "gemini"
        else:
            final_plan["decision_mode"] = "autonomous_heuristic"
            final_plan["llm_source"] = "deterministic"
        final_plan["reasoning_budget"] = {
            "planner": llm_gate.get("planner_reason", "Deterministic planner path was sufficient."),
            "budget_level": llm_gate.get("budget_level", "local_only"),
        }
        return final_plan

    def _build_analysis(self, state, demand, insight, event_context, operational_signals, recent_cycles, best_zone, learning_profile=None):
        most_crowded = min(state, key=lambda zone: state[zone]["free_slots"])
        congestion_threshold = self._dynamic_congestion_threshold(state, operational_signals)
        congested_zones = [
            zone for zone, data in state.items() if data["free_slots"] <= congestion_threshold
        ]
        
        blocked_routes = (learning_profile or {}).get("blocked_routes", [])
        
        recommended_zone = event_context.get("recommended_zone", best_zone)
        # Avoid recommended_zone if it's the source or blocked
        if recommended_zone == most_crowded or f"{most_crowded}->{recommended_zone}" in blocked_routes:
            recommended_zone = best_zone

        destination = recommended_zone if recommended_zone in state and recommended_zone != most_crowded and f"{most_crowded}->{recommended_zone}" not in blocked_routes else best_zone
        
        if destination == most_crowded or f"{most_crowded}->{destination}" in blocked_routes:
            destination = max(
                (zone for zone in state if zone != most_crowded and f"{most_crowded}->{zone}" not in blocked_routes),
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
        vehicles = max(0, min(safe_transfer_capacity, requested, state.get(destination_zone, {}).get("free_slots", 0)))
        confidence = max(
            0.45,
            min(
                0.95,
                float(insight.get("confidence", 0.7)) * uncertainty_discount + (0.06 if vehicles > 0 else -0.08),
            ),
        )

        # Removed stability gating: it caused confidence starvation and blocked valid actions.

        # Learning Integration: Check for recent failures and reward drift
        recent_reward_avg = learning_profile.get("recent_reward_avg", 0.0)
        last_reward = learning_profile.get("last_reward", 0.0)
        recent_failures = learning_profile.get("recent_failures", [])
        failure_count = sum(1 for f in recent_failures if f.get("from") == source_zone and f.get("to") == destination_zone)
        
        # Determine Learning Prefix
        learning_applied = False
        reason_prefix = ""
        if failure_count > 0:
            penalty = 0.15 * failure_count
            vehicles = max(0, int(vehicles * (1.0 - penalty)))
            confidence = max(0.4, confidence - 0.2)
            reason_prefix = f"🧠 Learning Applied: Reduced volume due to {failure_count} failures on this route. "
            learning_applied = True
        elif recent_reward_avg < -0.2:
            vehicles = max(1, int(vehicles * 0.7))
            reason_prefix = f"🧠 Learning Applied: Throttling traffic due to negative reward drift ({recent_reward_avg}). "
            learning_applied = True

        # Action Type Determination (Expansion)
        is_redirect = vehicles > 0 and source_zone != destination_zone and confidence >= 0.45
        
        # ── Step 4: "Do Nothing" (HOLD) Intelligence ──
        # If system is stable but last reward was bad, or avg reward is very bad, force a HOLD.
        is_stable = temporal_reasoning.get("pressure_trend") == "stable"
        force_hold = False
        if is_stable and last_reward < -0.4:
            force_hold = True
            reason_prefix = "🧠 Learning Applied: Hold strategy enforced. System is stable but previous action underperformed. "
        elif recent_reward_avg < -0.8:
            force_hold = True
            reason_prefix = "🧠 Learning Applied: Emergency Hold. Severe reward dip detected; halting all redirects to stabilize. "

        # Predictive Pre-Routing (PRE_ALLOCATE)
        # Act immediately if the trend shows increasing pressure
        is_pre_allocate = False
        projected_source_free = temporal_reasoning.get("projected_free_slots", {}).get(source_zone, source_free)
        if not is_redirect and not force_hold and temporal_reasoning.get("pressure_trend") in {"escalating", "elevated"} and source_zone != destination_zone:
            # Aggressive predictive move
            vehicles = max(1, min(12, int(demand.get(source_zone, 8) / 3)))
            vehicles = max(0, min(safe_transfer_capacity, vehicles, state.get(destination_zone, {}).get("free_slots", 0)))
            if vehicles > 0:
                is_pre_allocate = True
                is_redirect = True
                confidence = max(0.4, confidence - 0.05) # Slightly lower confidence for predictive moves

        # Confidence Gating
        if confidence < 0.55 and vehicles > 0 and not force_hold:
            vehicles = max(1, vehicles // 2)
            reason_prefix += f"Low confidence ({confidence:.2f}): reducing transfer volume for safety. "

        if is_redirect and not force_hold:
            action_type = "redirect"
            action_reason = (
                f"{reason_prefix}Proactive relief: Moving {vehicles} arrivals from {source_zone} to {destination_zone} "
                f"to prevent {temporal_reasoning['pressure_trend']} congestion."
            )
        elif is_pre_allocate and not force_hold:
            action_type = "redirect" # Environment only understands 'redirect'
            action_reason = (
                f"{reason_prefix}PRE-ALLOCATE: Source {source_zone} projected to hit critical levels ({projected_source_free} slots left) "
                f"in 2 cycles. Routing {vehicles} early to {destination_zone}."
            )
        else:
            action_type = "none"
            if force_hold:
                action_reason = reason_prefix
            else:
                action_reason = (
                    f"{reason_prefix}Monitoring: All zones within stable operating bounds. "
                    f"Projected {source_zone} free slots: {projected_source_free}."
                )

        return {
            "action": action_type,
            "is_pre_allocate": is_pre_allocate,
            "learning_applied": learning_applied or force_hold,
            "from": source_zone,
            "to": destination_zone,
            "vehicles": vehicles,
            "reason": action_reason,
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
        learning_profile=None,
    ):
        blocked_zone = operational_signals.get("blocked_zone")
        blocked_routes = (learning_profile or {}).get("blocked_routes", [])
        candidates = [
            zone for zone in state
            if zone != source_zone and zone != blocked_zone and f"{source_zone}->{zone}" not in blocked_routes
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

    def _score_plan(self, action, analysis, temporal_reasoning, insight, safe_transfer_capacity, adaptive_limits, learning_profile=None):
        vehicles = max(0, int(action.get("vehicles", 0) or 0))
        capacity_score = min(1.0, vehicles / max(1, safe_transfer_capacity)) if safe_transfer_capacity else 0.0
        hotspot_relief = min(
            1.0,
            max(0.0, (adaptive_limits["congestion_threshold"] + 4 - analysis["queue_length"]) / max(1, adaptive_limits["congestion_threshold"] + 4)),
        )
        projected_pressure = temporal_reasoning["projected_queue_peak"] / 6
        uncertainty = min(1.0, float(insight.get("uncertainty", {}).get("entropy", 0.0)) / 3)
        
        # Calculate Learning Penalty
        failure_penalty = 0.0
        if learning_profile and action.get("from") and action.get("to"):
            route_key = f"{action.get('from')}->{action.get('to')}"
            failure_penalty = self._get_failure_penalty(route_key, learning_profile)

        benefit_score = round(
            min(1.0, 0.45 * capacity_score + 0.35 * hotspot_relief + 0.20 * min(1.0, projected_pressure + 0.2)),
            3,
        )
        risk_probability = round(
            min(0.99, 0.20 + uncertainty * 0.35 + max(0, analysis["queue_length"] - 2) * 0.06 + failure_penalty),
            3,
        )
        efficiency_score = round(max(0.0, min(1.0, benefit_score - risk_probability * 0.35 + 0.15 - (failure_penalty * 0.5))), 3)
        confidence = round(max(0.35, min(0.95, action.get("confidence", 0.65) - risk_probability * 0.12 - failure_penalty)), 3)
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
        
        # Predictive projection: how many slots will be free in 2 steps?
        projected_free_slots = {}
        for zone, data in state.items():
            current_free = data.get("free_slots", 0)
            inflow = demand.get(zone, 0)
            expected_change = int(inflow * 2.2) # Weighting inflow for 2 steps
            projected_free_slots[zone] = max(0, current_free - expected_change)

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
            "projected_free_slots": projected_free_slots,
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

    def _get_failure_penalty(self, route_key, learning_profile):
        """Calculate a numerical penalty score based on route-specific learning history."""
        if route_key in learning_profile.get("blocked_routes", []):
            return 0.55  # Critical penalty for formally blocked routes

        consecutive_failures = learning_profile.get("route_consecutive_failures", {}).get(route_key, 0)
        
        if consecutive_failures > 5:
            return 0.35
        elif consecutive_failures > 2:
            return 0.18
        elif consecutive_failures > 0:
            return 0.08
            
        return 0.0

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
        fallback_action = merged["proposed_action"]
        llm_action = llm_advisory.get("proposed_action", {})
        if self._is_safe_llm_action(fallback_action, llm_action, safe_transfer_capacity):
            # Check if LLM ACTUALLY changed the action vs baseline
            if (llm_action.get("action") != fallback_action.get("action") or
                llm_action.get("from") != fallback_action.get("from") or
                llm_action.get("to") != fallback_action.get("to") or
                llm_action.get("vehicles") != fallback_action.get("vehicles")):
                merged["llm_influence"] = True
            merged["proposed_action"] = llm_action
            merged["llm_advisory_used"] = True
        if isinstance(llm_advisory.get("alternative_actions"), list) and llm_advisory["alternative_actions"]:
            merged["alternative_actions"] = llm_advisory["alternative_actions"][:3]
            merged["llm_advisory_used"] = True
        if isinstance(llm_advisory.get("strategy"), str) and llm_advisory["strategy"].strip():
            merged["strategy"] = llm_advisory["strategy"].strip()
        if isinstance(llm_advisory.get("rationale"), str) and llm_advisory["rationale"].strip():
            merged["rationale"] = llm_advisory["rationale"].strip()
            merged["llm_summary"] = merged["rationale"]
        return merged

    def _is_safe_llm_action(self, fallback_action, llm_action, safe_transfer_capacity):
        if not isinstance(llm_action, dict):
            return False
        if fallback_action.get("action") == "none":
            # Allow LLM to actively initiate a redirect even when fallback is none
            if llm_action.get("action") == "redirect" and 0 < int(llm_action.get("vehicles", 0) or 0) <= safe_transfer_capacity:
                return True
            return llm_action.get("action") == "none"
        if llm_action.get("action") == "none":
            return True
        return (
            llm_action.get("action") == "redirect"
            and 0 < int(llm_action.get("vehicles", 0) or 0) <= safe_transfer_capacity
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
