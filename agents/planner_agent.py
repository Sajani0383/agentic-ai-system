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
        belief_state = self._call_tool(
            tools,
            "build_belief_state",
            state,
            demand,
            insight,
            default={},
            tool_calls=tool_calls,
        )
        reward_trend = self._call_tool(
            tools,
            "reward_trend_analysis",
            default={},
            tool_calls=tool_calls,
        )
        tool_observations["belief_state"] = belief_state
        tool_observations["reward_trend"] = reward_trend

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
            other_zones = [
                z for z in state
                if z != source_zone
                and z != destination_zone
                and z not in recently_failed_zones
                and f"{source_zone}->{z}" not in blocked_routes
            ]
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
        adaptive_limits = self._build_adaptive_limits(state, demand, event_context, operational_signals, insight, learning_profile, goal_status=goal_status)
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
        primary_action, short_horizon_eval = self._select_best_short_horizon_action(
            primary_action,
            alternative_actions,
            state,
            demand,
            temporal_reasoning,
            adaptive_limits,
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
            "belief_state": belief_state,
            "short_horizon_eval": short_horizon_eval,
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
        fallback_plan["local_decision_snapshot"] = deepcopy(fallback_plan["proposed_action"])

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

        llm_runtime = llm_advisory.get("_llm_runtime", {}) if isinstance(llm_advisory, dict) else {}
        llm_decision_snapshot = deepcopy(llm_advisory.get("proposed_action", {})) if isinstance(llm_advisory, dict) else {}
        final_plan = self._merge_llm_advisory(fallback_plan, llm_advisory, safe_transfer_capacity)
        final_plan = self._finalize_plan_action(
            final_plan,
            state,
            demand,
            learning_profile,
            temporal_reasoning,
            adaptive_limits,
            safe_transfer_capacity,
        )
        final_plan.setdefault("tool_calls", tool_calls)
        final_plan.setdefault("tool_observations", tool_observations)
        final_plan.setdefault("adaptive_limits", adaptive_limits)
        final_plan.setdefault("alternative_actions", alternative_actions)
        final_plan.setdefault("action_sequence", action_sequence)
        final_plan.setdefault("scoring", scoring)
        final_plan.setdefault("temporal_reasoning", temporal_reasoning)
        final_plan.setdefault("planner_feedback", planner_feedback)
        final_plan["llm_requested"] = llm_requested
        final_plan["llm_runtime"] = llm_runtime
        final_plan["llm_fallback_used"] = bool(llm_runtime.get("fallback_used"))
        final_plan["llm_response_used"] = bool(llm_runtime.get("used"))
        final_plan["llm_error"] = llm_runtime.get("error", "")
        final_plan["llm_fallback_reason"] = llm_runtime.get("fallback_reason", "")
        final_plan["forced_live_attempt"] = bool(llm_gate.get("force_llm", False) and llm_strategy == "gemini")
        final_plan["local_decision_snapshot"] = deepcopy(fallback_plan.get("proposed_action", {}))
        final_plan["llm_decision_snapshot"] = llm_decision_snapshot
        final_plan["final_decision_snapshot"] = deepcopy(final_plan.get("proposed_action", {}))
        live_gemini_used = bool(llm_strategy == "gemini" and final_plan.get("llm_response_used") and not final_plan.get("llm_fallback_used"))
        if live_gemini_used:
            final_plan["llm_decision_status"] = "modified" if final_plan.get("llm_influence") else "confirmed"
        elif llm_strategy == "gemini" and llm_requested and final_plan.get("llm_fallback_used"):
            final_plan["llm_decision_status"] = "fallback"
        elif llm_strategy in {"cached", "demo_simulated"}:
            final_plan["llm_decision_status"] = "modified" if final_plan.get("llm_influence") else "reused"
        elif llm_strategy == "local_simulated":
            final_plan["llm_decision_status"] = "simulated"
        else:
            final_plan["llm_decision_status"] = "local"
        if llm_strategy == "cached":
            final_plan["decision_mode"] = "cached_llm_advisory"
            final_plan["llm_source"] = "cached"
            final_plan["llm_advisory_used"] = True
        elif llm_strategy == "local_simulated":
            final_plan["decision_mode"] = "local_ai_simulation"
            final_plan["llm_source"] = "local_simulated"
            final_plan["llm_advisory_used"] = False
        elif llm_strategy == "demo_simulated":
            final_plan["decision_mode"] = "demo_simulated_gemini"
            final_plan["llm_source"] = "demo_simulated"
            final_plan["llm_advisory_used"] = True
        elif llm_strategy == "gemini" and llm_requested and final_plan.get("llm_fallback_used"):
            final_plan["decision_mode"] = "autonomous_localized_fallback"
            final_plan["llm_source"] = "gemini_failed_fallback"
            final_plan["llm_advisory_used"] = False
        elif live_gemini_used or final_plan.get("llm_advisory_used"):
            final_plan["decision_mode"] = "distributed_cloud_briefing"
            final_plan["llm_source"] = "gemini"
            final_plan["llm_advisory_used"] = True
        else:
            final_plan["decision_mode"] = "autonomous_heuristic"
            final_plan["llm_source"] = "deterministic"
        final_plan["reasoning_budget"] = {
            "planner": llm_gate.get("planner_reason", "Deterministic planner path was sufficient."),
            "budget_level": llm_gate.get("budget_level", "local_only"),
        }
        return final_plan

    def _finalize_plan_action(
        self,
        plan,
        state,
        demand,
        learning_profile,
        temporal_reasoning,
        adaptive_limits,
        safe_transfer_capacity,
    ):
        finalized = deepcopy(plan)
        action = dict(finalized.get("proposed_action", {"action": "none"}))
        blocked_routes = learning_profile.get("blocked_routes", [])
        route_key = f"{action.get('from')}->{action.get('to')}"

        if action.get("action") == "redirect" and route_key in blocked_routes:
            alternative = self._select_best_destination(
                state,
                demand,
                action.get("from"),
                action.get("to"),
                blocked_routes,
                temporal_reasoning,
                adaptive_limits,
                learning_profile,
            )
            if alternative:
                action["to"] = alternative
                micro_cap = 2 + (int((insight or {}).get("queue_length", 0) or demand.get(action.get("from"), 0) or 0) % 4)
                action["vehicles"] = max(1, min(micro_cap, safe_transfer_capacity or micro_cap, state.get(alternative, {}).get("free_slots", 0)))
                action["reason"] = f"Memory hard block avoided {route_key}; rerouting micro-action to {alternative}."
                action["force_micro"] = True
            else:
                action = {"action": "none", "reason": f"Memory hard block rejected {route_key}; no safe alternative route is open."}

        if action.get("action") == "redirect":
            confidence = float(action.get("confidence", 0.65) or 0.0)
            if confidence <= 0 and int(action.get("vehicles", 0) or 0) > 1:
                action["vehicles"] = 1
                action["force_micro"] = True
                action["reason"] = f"{action.get('reason', '')} Low confidence forced a one-vehicle micro-action.".strip()

        if action.get("action") == "none" and bool(learning_profile.get("none_block_active", False)):
            source_zone = min(state, key=lambda zone: state[zone].get("free_slots", 0)) if state else None
            destination_zone = self._select_best_destination(
                state,
                demand,
                source_zone,
                None,
                blocked_routes,
                temporal_reasoning,
                adaptive_limits,
                learning_profile,
            )
            if source_zone and destination_zone:
                micro_cap = 2 + (int((insight or {}).get("queue_length", 0) or demand.get(source_zone, 0) or 0) % 4)
                vehicles = max(1, min(micro_cap, safe_transfer_capacity or micro_cap, state[destination_zone].get("free_slots", 0)))
                action = {
                    "action": "redirect",
                    "from": source_zone,
                    "to": destination_zone,
                    "vehicles": vehicles,
                    "reason": "Learning hard block: repeated NONE failures require a small recovery redirect.",
                    "confidence": 0.45,
                    "force_micro": True,
                    "expected_gain": 0.2,
                    "next_step_effect": {"improvement": vehicles},
                }

        finalized["proposed_action"] = action
        if finalized.get("action_sequence"):
            finalized["action_sequence"][0]["action"] = deepcopy(action)
        return finalized

    def _build_analysis(self, state, demand, insight, event_context, operational_signals, recent_cycles, best_zone, learning_profile=None):
        most_crowded = min(state, key=lambda zone: state[zone]["free_slots"])
        congestion_threshold = self._dynamic_congestion_threshold(state, operational_signals)
        congested_zones = [
            zone for zone, data in state.items() if data["free_slots"] <= congestion_threshold
        ]
        
        blocked_routes = list((learning_profile or {}).get("blocked_routes", []))
        for rule in (learning_profile or {}).get("llm_memory_rules", []):
            if not isinstance(rule, dict):
                continue
            if float(rule.get("strength", 0.0) or 0.0) < -0.35:
                route_key = rule.get("route_key") or f"{rule.get('from')}->{rule.get('to')}"
                if route_key not in blocked_routes:
                    blocked_routes.append(route_key)
        
        recommended_zone = event_context.get("recommended_zone", best_zone)
        # Avoid recommended_zone if it's the source or blocked
        if recommended_zone == most_crowded or f"{most_crowded}->{recommended_zone}" in blocked_routes:
            recommended_zone = best_zone

        destination = recommended_zone if recommended_zone in state and recommended_zone != most_crowded and f"{most_crowded}->{recommended_zone}" not in blocked_routes else best_zone
        llm_rules = (learning_profile or {}).get("llm_memory_rules", [])
        preferred = self._select_llm_preferred_destination(llm_rules, most_crowded, state, blocked_routes)
        if preferred:
            destination = preferred
        
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
            "blocked_routes": blocked_routes[:],
            "llm_memory_rules_applied": bool(preferred),
        }

    def _select_llm_preferred_destination(self, llm_rules, source_zone, state, blocked_routes):
        if not isinstance(llm_rules, list):
            return None
        candidates = []
        for rule in llm_rules:
            if not isinstance(rule, dict):
                continue
            destination = rule.get("to")
            strength = float(rule.get("strength", 0.0) or 0.0)
            if (
                rule.get("from") == source_zone
                and destination in state
                and destination != source_zone
                and f"{source_zone}->{destination}" not in blocked_routes
                and state[destination].get("free_slots", 0) > 0
                and strength > 0
            ):
                candidates.append((strength, state[destination].get("free_slots", 0), destination))
        if not candidates:
            return None
        return max(candidates)[2]

    def _llm_rule_strength(self, learning_profile, source_zone, destination_zone):
        if not learning_profile or not source_zone or not destination_zone:
            return 0.0
        rules = learning_profile.get("llm_memory_rules", [])
        strength = 0.0
        for rule in rules if isinstance(rules, list) else []:
            if rule.get("from") == source_zone and rule.get("to") == destination_zone:
                strength += float(rule.get("strength", 0.0) or 0.0)
        return round(strength, 3)

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

    def _build_adaptive_limits(self, state, demand, event_context, operational_signals, insight, learning_profile, goal_status=None):
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
        # Goal-alignment bonus: increase bias when actions are aligned with improving worst zone
        goal_zone = goal_status.get("target_zone") if goal_status else None
        goal_bonus = 1.1 if goal_zone and goal_zone == source_zone else 1.0
        # Policy dominance: scale max_transfer directly from last_policy_reward
        last_policy_reward = float(learning_profile.get("last_policy_reward", 0.0))
        policy_weight = max(0.6, min(1.5, 1.0 + last_policy_reward))
        max_transfer = max(4, min(24, int(max_transfer * policy_weight * goal_bonus)))
        return {
            "max_transfer": max_transfer,
            "congestion_threshold": congestion_threshold,
            "horizon_steps": horizon_steps,
            "learned_bias": round(learned_bias, 3),
            "source_pressure": source_pressure,
            "uncertainty_penalty": round(min(0.25, float(insight.get("uncertainty", {}).get("entropy", 0.0)) / 10), 3),
            "policy_weight": round(policy_weight, 3),
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
        blocked_routes = learning_profile.get("blocked_routes", [])
        queue_pressure = temporal_reasoning.get("projected_queue_peak", 0)
        force_micro_action = (
            recent_reward_avg < -0.1
            or queue_pressure >= 2
            or bool(learning_profile.get("force_recovery_redirect_next_step", False))
            or bool(learning_profile.get("none_block_active", False))
        )
        blocked_route = f"{source_zone}->{destination_zone}" in blocked_routes
        
        # Determine Learning Prefix
        learning_applied = False
        reason_prefix = ""
        if blocked_route:
            alternative = self._select_best_destination(
                state,
                demand,
                source_zone,
                destination_zone,
                blocked_routes,
                temporal_reasoning,
                adaptive_limits,
                learning_profile,
            )
            if alternative:
                reason_prefix = f"Memory Hard Block: Avoiding blocked route {source_zone}->{destination_zone}; using {alternative}. "
                destination_zone = alternative
                destination_free = state.get(destination_zone, {}).get("free_slots", 0)
                safe_transfer_capacity = min(safe_transfer_capacity or adaptive_limits["max_transfer"], destination_free)
                vehicles = max(0, min(vehicles or 2, safe_transfer_capacity))
                learning_applied = True
            else:
                vehicles = 0
                reason_prefix = f"Memory Hard Block: Route {source_zone}->{destination_zone} is blocked and no safe alternative is open. "
                learning_applied = True
        elif failure_count > 0:
            penalty = 0.15 * failure_count
            vehicles = max(0, int(vehicles * (1.0 - penalty)))
            confidence = max(0.4, confidence - 0.2)
            reason_prefix = f"🧠 Learning Applied: Reduced volume due to {failure_count} failures on this route. "
            learning_applied = True

        # Failure recovery: If the current route has repeated failures, switch to best alternative
        route_consecutive_failures = learning_profile.get("route_consecutive_failures", {})
        current_route_failures = route_consecutive_failures.get(f"{source_zone}->{destination_zone}", 0)
        if current_route_failures >= 2 and source_zone != destination_zone:
            # Find best alternative route not in blocked list
            blocked_routes_local = learning_profile.get("blocked_routes", [])
            alt_zones = [z for z in state if z != source_zone and z != destination_zone
                         and f"{source_zone}->{z}" not in blocked_routes_local]
            if alt_zones:
                alt_best = max(alt_zones, key=lambda z: state[z].get("free_slots", 0))
                reason_prefix = f"🔄 Failure Recovery: Route {source_zone}->{destination_zone} failed {current_route_failures}x. Switching to {alt_best}. "
                destination_zone = alt_best
                learning_applied = True

        llm_rule_strength = self._llm_rule_strength(learning_profile, source_zone, destination_zone)
        if llm_rule_strength <= -0.35 and source_zone != destination_zone:
            alternative = self._select_best_destination(
                state,
                demand,
                source_zone,
                destination_zone,
                blocked_routes,
                temporal_reasoning,
                adaptive_limits,
                learning_profile,
            )
            if alternative and alternative != destination_zone:
                reason_prefix = (
                    f"LLM memory veto: route {source_zone}->{destination_zone} carried negative learned signal "
                    f"({llm_rule_strength:.2f}); switching to {alternative}. "
                )
                destination_zone = alternative
                destination_free = state.get(destination_zone, {}).get("free_slots", 0)
                safe_transfer_capacity = min(safe_transfer_capacity or adaptive_limits["max_transfer"], destination_free)
                vehicles = max(0, min(max(1, vehicles), safe_transfer_capacity))
                llm_rule_strength = self._llm_rule_strength(learning_profile, source_zone, destination_zone)
                learning_applied = True
            else:
                vehicles = min(vehicles, 1)
                reason_prefix += (
                    f"LLM memory caution: route {source_zone}->{destination_zone} has negative prior signal "
                    f"({llm_rule_strength:.2f}), so transfer stays micro. "
                )
                learning_applied = True
        elif llm_rule_strength >= 0.35 and source_zone != destination_zone and vehicles > 0:
            llm_volume_boost = 2 if llm_rule_strength >= 0.75 else 1
            boosted = min(safe_transfer_capacity, state.get(destination_zone, {}).get("free_slots", 0), vehicles + llm_volume_boost)
            if boosted > vehicles:
                vehicles = boosted
                confidence = min(0.95, confidence + 0.05)
                reason_prefix += (
                    f"LLM memory reinforcement: prior success on {source_zone}->{destination_zone} "
                    f"supports a stronger redirect ({llm_rule_strength:.2f}). "
                )
                learning_applied = True

        if recent_reward_avg < -0.1:
            alternative = self._select_best_destination(
                state,
                demand,
                source_zone,
                destination_zone,
                blocked_routes,
                temporal_reasoning,
                adaptive_limits,
                learning_profile,
            )
            if alternative and alternative != destination_zone and learning_profile.get("route_profile", {}).get("avg_reward", 0.0) < 0:
                destination_zone = alternative
                destination_free = state.get(destination_zone, {}).get("free_slots", 0)
                safe_transfer_capacity = min(safe_transfer_capacity or adaptive_limits["max_transfer"], destination_free)
            micro_cap = 2 + (int((insight or {}).get("queue_length", 0) or demand.get(source_zone, 0) or 0) % 4)
            vehicles = max(1, min(micro_cap if force_micro_action else max(1, int(vehicles * 0.7)), safe_transfer_capacity))
            reason_prefix = f"🧠 Learning Applied: Throttling traffic due to negative reward drift ({recent_reward_avg}). "
            learning_applied = True
        elif recent_reward_avg > 0.1 and vehicles > 0:
            vehicles = min(safe_transfer_capacity, state.get(destination_zone, {}).get("free_slots", 0), vehicles + (2 if learning_profile.get("route_profile", {}).get("success_bias", 1.0) >= 1.15 else 1))
            reason_prefix = "Reward trend improving: allowing a slightly larger safe redirect. "
            learning_applied = True

        # Action Type Determination (Expansion)
        is_redirect = vehicles > 0 and source_zone != destination_zone and confidence >= 0.45
        
        # ── Step 4: "Do Nothing" (HOLD) Intelligence ──
        is_stable = temporal_reasoning.get("pressure_trend") == "stable"
        force_hold = bool(learning_profile.get("force_hold_next_step", False))  # Severe-reward override from controller
        if force_hold and force_micro_action:
            force_hold = False
            micro_capacity = min(safe_transfer_capacity, state.get(destination_zone, {}).get("free_slots", 0))
            vehicles = max(0, min(2, micro_capacity))
            reason_prefix = "Reward recovery action: replacing repeated hold with a small safe redirect. "
            is_redirect = vehicles > 0 and source_zone != destination_zone
        elif force_hold:
            reason_prefix = "Reward guardrail: pausing redirects for one recovery cycle. "
        elif is_stable and last_reward < -0.4 and not force_micro_action:
            force_hold = True
            reason_prefix = "🧠 Learning Applied: Hold strategy enforced. System is stable but previous action underperformed. "
        elif recent_reward_avg < -0.8 and not force_micro_action:
            force_hold = True
            reason_prefix = "🧠 Learning Applied: Emergency Hold. Severe reward dip detected; halting all redirects to stabilize. "

        # Predictive Pre-Routing (PRE_ALLOCATE)
        # Act immediately if the trend shows increasing pressure
        is_pre_allocate = False
        projected_source_free = temporal_reasoning.get("projected_free_slots", {}).get(source_zone, source_free)
        if not is_redirect and not force_hold and temporal_reasoning.get("pressure_trend") in {"escalating", "elevated"} and source_zone != destination_zone:
            # Aggressive predictive move
            vehicles = max(1, min(4, int(demand.get(source_zone, 8) / 6)))
            vehicles = max(0, min(safe_transfer_capacity, vehicles, state.get(destination_zone, {}).get("free_slots", 0)))
            expected_gain = self._estimate_expected_gain(
                state,
                demand,
                source_zone,
                destination_zone,
                vehicles,
                temporal_reasoning,
                adaptive_limits,
            )
            if vehicles > 0 and expected_gain > 0.15:
                is_pre_allocate = True
                is_redirect = True
                confidence = max(0.4, confidence - 0.05) # Slightly lower confidence for predictive moves

        expected_gain = self._estimate_expected_gain(
            state,
            demand,
            source_zone,
            destination_zone,
            vehicles,
            temporal_reasoning,
            adaptive_limits,
        )
        next_step_effect = self._estimate_next_step_effect(
            state,
            source_zone,
            destination_zone,
            vehicles,
            adaptive_limits,
        )
        if force_micro_action and not is_redirect and not force_hold and source_zone != destination_zone:
            micro_capacity = min(safe_transfer_capacity, state.get(destination_zone, {}).get("free_slots", 0))
            vehicles = max(0, min(2, micro_capacity))
            expected_gain = self._estimate_expected_gain(state, demand, source_zone, destination_zone, vehicles, temporal_reasoning, adaptive_limits)
            next_step_effect = self._estimate_next_step_effect(state, source_zone, destination_zone, vehicles, adaptive_limits)
            if expected_gain > 0.15 and next_step_effect["improvement"] > 0:
                is_redirect = True
                reason_prefix += "Decisive recovery: queue/reward pressure triggered a small safe redirect. "

        # Confidence Gating
        if confidence < 0.55 and vehicles > 0 and not force_hold:
            vehicles = max(1, vehicles // 2)
            reason_prefix += f"Low confidence ({confidence:.2f}): reducing transfer volume for safety. "
            expected_gain = self._estimate_expected_gain(state, demand, source_zone, destination_zone, vehicles, temporal_reasoning, adaptive_limits)
            next_step_effect = self._estimate_next_step_effect(state, source_zone, destination_zone, vehicles, adaptive_limits)

        if is_redirect and (expected_gain <= 0.15 or next_step_effect["improvement"] <= 0) and not force_micro_action:
            is_redirect = False
            vehicles = 0
            reason_prefix += "Quality filter: expected gain was negligible, so redirect was cancelled. "
        elif is_redirect and force_micro_action and expected_gain <= 0.15:
            expected_gain = max(expected_gain, 0.16)
            reason_prefix += "Learning pressure override: retaining micro-action despite low estimated gain. "

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

        # Multi-step lookahead: weight the action score by 3-step projected pressure delta
        route_profiles = learning_profile.get("route_profile", {})
        route_historical_score = route_profiles.get("avg_reward", 0.0)
        multi_step_weight = 1.0 + max(-0.3, min(0.3, route_historical_score))  # ±30% from historical data
        if action_type == "redirect" and multi_step_weight != 1.0:
            vehicles = max(1, int(vehicles * multi_step_weight))
            vehicles = min(vehicles, safe_transfer_capacity)
            expected_gain = self._estimate_expected_gain(state, demand, source_zone, destination_zone, vehicles, temporal_reasoning, adaptive_limits)
            next_step_effect = self._estimate_next_step_effect(state, source_zone, destination_zone, vehicles, adaptive_limits)

        return {
            "action": action_type,
            "is_pre_allocate": is_pre_allocate,
            "learning_applied": learning_applied or force_hold,
            "from": source_zone,
            "to": destination_zone,
            "vehicles": vehicles,
            "reason": action_reason,
            "confidence": round(confidence, 3),
            "multi_step_weight": round(multi_step_weight, 3),
            "expected_gain": round(expected_gain, 3),
            "next_step_effect": next_step_effect,
        }

    def _select_best_destination(self, state, demand, source_zone, current_destination, blocked_routes, temporal_reasoning, adaptive_limits, learning_profile=None):
        candidates = [
            zone for zone in state
            if zone != source_zone and f"{source_zone}->{zone}" not in blocked_routes
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda zone: (
                self._estimate_expected_gain(
                    state,
                    demand,
                    source_zone,
                    zone,
                    max(0, min(2, state[zone].get("free_slots", 0))),
                    temporal_reasoning,
                    adaptive_limits,
                )
                + self._llm_rule_bias(learning_profile=learning_profile, source_zone=source_zone, destination_zone=zone)
            ),
        )

    def _llm_rule_bias(self, learning_profile, source_zone, destination_zone):
        return self._llm_rule_strength(learning_profile, source_zone, destination_zone) * 0.35

    def _estimate_expected_gain(self, state, demand, source_zone, destination_zone, vehicles, temporal_reasoning, adaptive_limits):
        if vehicles <= 0 or source_zone not in state or destination_zone not in state or source_zone == destination_zone:
            return 0.0
        source_free = state[source_zone].get("free_slots", 0)
        destination_free = state[destination_zone].get("free_slots", 0)
        source_deficit = max(0, adaptive_limits["congestion_threshold"] - source_free)
        demand_delta = max(0, demand.get(source_zone, 0) - demand.get(destination_zone, 0))
        queue_gain = max(0, temporal_reasoning.get("projected_queue_peak", 0) - 2) * 0.2
        relief_gain = min(vehicles, max(1, source_deficit + 1)) * 0.45
        destination_penalty = 0.4 if destination_free <= vehicles + 2 else 0.0
        return round(max(0.0, relief_gain + source_deficit * 0.08 + demand_delta * 0.02 + queue_gain - destination_penalty), 3)

    def _estimate_next_step_effect(self, state, source_zone, destination_zone, vehicles, adaptive_limits):
        if vehicles <= 0 or source_zone not in state or destination_zone not in state:
            return {"improvement": 0, "source_free_after": state.get(source_zone, {}).get("free_slots", 0), "destination_free_after": state.get(destination_zone, {}).get("free_slots", 0)}
        source_after = state[source_zone].get("free_slots", 0) + vehicles
        destination_after = state[destination_zone].get("free_slots", 0) - vehicles
        threshold = adaptive_limits["congestion_threshold"]
        before_deficit = max(0, threshold - state[source_zone].get("free_slots", 0))
        after_deficit = max(0, threshold - source_after)
        improvement = before_deficit - after_deficit
        if improvement <= 0 and state[source_zone].get("free_slots", 0) < state[destination_zone].get("free_slots", 0):
            improvement = min(vehicles, state[destination_zone].get("free_slots", 0) - state[source_zone].get("free_slots", 0)) * 0.2
        if destination_after < max(2, threshold // 2):
            improvement -= 1
        return {
            "improvement": round(improvement, 3),
            "source_free_after": source_after,
            "destination_free_after": destination_after,
        }

    def _select_best_short_horizon_action(
        self,
        primary_action,
        alternatives,
        state,
        demand,
        temporal_reasoning,
        adaptive_limits,
        learning_profile,
    ):
        candidates = [primary_action] + [action for action in alternatives if isinstance(action, dict)]
        blocked_routes = set(learning_profile.get("blocked_routes", []))
        plan_patterns = learning_profile.get("plan_patterns", {})
        evaluations = []
        for candidate in candidates:
            action = deepcopy(candidate)
            if action.get("action") != "redirect":
                score = -0.15 if learning_profile.get("none_block_active") else 0.0
                evaluations.append({"action": action, "score": score, "reason": "Hold action only preserves current state."})
                continue
            route_key = f"{action.get('from')}->{action.get('to')}"
            if route_key in blocked_routes:
                evaluations.append({"action": action, "score": -99.0, "reason": "Route is hard-blocked by memory."})
                continue
            vehicles = max(0, int(action.get("vehicles", 0) or 0))
            expected_gain = self._estimate_expected_gain(
                state,
                demand,
                action.get("from"),
                action.get("to"),
                vehicles,
                temporal_reasoning,
                adaptive_limits,
            )
            next_step_effect = self._estimate_next_step_effect(
                state,
                action.get("from"),
                action.get("to"),
                vehicles,
                adaptive_limits,
            )
            pattern_key = f"{route_key}:{'micro' if vehicles <= 2 else 'standard'}"
            pattern_reward = float(plan_patterns.get(pattern_key, {}).get("avg_reward", 0.0) or 0.0)
            destination_after = next_step_effect.get("destination_free_after", state.get(action.get("to"), {}).get("free_slots", 0))
            capacity_penalty = 0.4 if destination_after < 2 else 0.0
            score = round(expected_gain + next_step_effect.get("improvement", 0.0) * 0.35 + pattern_reward * 0.25 - capacity_penalty, 3)
            score += self._llm_rule_bias(
                learning_profile=learning_profile,
                source_zone=action.get("from"),
                destination_zone=action.get("to"),
            )
            evaluations.append(
                {
                    "action": action,
                    "score": score,
                    "expected_gain": expected_gain,
                    "next_step_effect": next_step_effect,
                    "pattern_reward": pattern_reward,
                    "reason": "Two-step relief estimate with memory pattern reward.",
                }
            )
        best = max(evaluations, key=lambda item: item["score"]) if evaluations else {"action": primary_action, "score": 0.0}
        selected = deepcopy(best["action"])
        primary_score = evaluations[0]["score"] if evaluations else 0.0
        if best["score"] > primary_score + 0.1:
            selected["reason"] = f"{selected.get('reason', '')} Short-horizon evaluator selected this route over the initial plan.".strip()
            selected["short_horizon_selected"] = True
        return selected, {
            "selected_score": best["score"],
            "primary_score": primary_score,
            "evaluations": [
                {
                    "action": item["action"],
                    "score": item["score"],
                    "reason": item.get("reason", ""),
                }
                for item in evaluations[:4]
            ],
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
            "expected_gain": action.get("expected_gain", benefit_score),
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
        plan_patterns = learning_profile.get("plan_patterns", {})
        micro_pattern = plan_patterns.get(f"{route_key}:micro", {})
        standard_pattern = plan_patterns.get(f"{route_key}:standard", {})
        pattern_reward = min(
            float(micro_pattern.get("avg_reward", 0.0) or 0.0),
            float(standard_pattern.get("avg_reward", 0.0) or 0.0),
        ) if micro_pattern or standard_pattern else 0.0
        
        if consecutive_failures > 5:
            return 0.35
        elif consecutive_failures > 2:
            return 0.18
        elif consecutive_failures > 0:
            return 0.08
        elif pattern_reward < -0.2:
            return min(0.3, abs(pattern_reward) * 0.45)
            
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
