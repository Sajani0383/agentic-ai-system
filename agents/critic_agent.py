from copy import deepcopy

from adk.trace_logger import trace_logger
from llm_reasoning import ask_llm_for_structured_json


class CriticAgent:
    def __init__(self, logger=None):
        self.logger = logger or trace_logger
        self.route_risk_memory = {}

    def review(self, plan, state, demand, insight, tools, reasoning_budget=None):
        proposed_action = plan.get("proposed_action", {"action": "none"})
        tool_calls = list(plan.get("tool_calls", []))
        event_context = self._call_tool(
            tools,
            "get_event_context",
            default={},
            tool_calls=tool_calls,
        )
        scenario_mode = self._call_tool(
            tools,
            "get_scenario_mode",
            default="Unknown",
            tool_calls=tool_calls,
        )
        operational_signals = self._call_tool(
            tools,
            "get_operational_signals",
            default={},
            tool_calls=tool_calls,
        )
        learning_profile = self._call_tool(
            tools,
            "get_learning_profile",
            default={},
            keyword_args={
                "scenario_mode": scenario_mode,
                "from_zone": proposed_action.get("from"),
                "to_zone": proposed_action.get("to"),
            },
            tool_calls=tool_calls,
        )

        deterministic_review = self._build_deterministic_review(
            proposed_action,
            state,
            demand,
            insight,
            plan,
            tools,
            event_context,
            operational_signals,
            learning_profile,
            tool_calls,
        )
        llm_gate = reasoning_budget or {}
        llm_requested = bool(llm_gate.get("allow_critic_llm"))
        llm_review = deterministic_review
        if llm_requested:
            llm_review = self._ask_llm_for_review(
                deterministic_review,
                plan,
                state,
                demand,
                insight,
                event_context,
                operational_signals,
                learning_profile,
                scenario_mode,
            )
        review = self._merge_llm_review(deterministic_review, llm_review)
        review["llm_requested"] = llm_requested
        review["reasoning_budget"] = {
            "critic": llm_gate.get("critic_reason", "Deterministic critic checks were sufficient."),
            "budget_level": llm_gate.get("budget_level", "local_only"),
        }
        self._update_critic_memory(review)
        self._log_review(review)
        return review

    def _build_deterministic_review(
        self,
        action,
        state,
        demand,
        insight,
        plan,
        tools,
        event_context,
        operational_signals,
        learning_profile,
        tool_calls,
    ):
        notes = []
        scoring = self._score_action(
            action,
            state,
            demand,
            insight,
            tools,
            event_context,
            operational_signals,
            learning_profile,
            tool_calls,
        )
        revised_action = dict(action)
        safe_transfer = scoring["observations"].get("safe_transfer", 0)
        from_zone = action.get("from")
        to_zone = action.get("to")

        # ── Negligible Benefit Veto ──
        source_data = state.get(from_zone, {})
        dest_data = state.get(to_zone, {})
        source_free_pct = (source_data.get("free_slots", 0) / source_data.get("total_slots", 24)) * 100 if from_zone in state else 0
        dest_free_pct = (dest_data.get("free_slots", 0) / dest_data.get("total_slots", 24)) * 100 if to_zone in state else 0

        is_negligible = bool(scoring.get("risk_factors", {}).get("low_utility_penalty", 0) > 0)
        queue_length_val = operational_signals.get("queue_length", 0)
        if queue_length_val > 2:
            is_negligible = False

        if action.get("action") == "redirect" and not is_negligible:
            # Secondary check for relative distribution stability
            if source_free_pct > 35 and dest_free_pct > 80:
                is_negligible = True
                if queue_length_val <= 2:
                    notes.append(f"VETO: Negligible benefit. {from_zone} is already {source_free_pct:.0f}% free. Redirection to {to_zone} is unnecessary movement.")
                else:
                    is_negligible = False

        # Allow safe micro-actions to explore and act
        if action.get("action") == "redirect" and int(action.get("vehicles", 0)) == 1 and scoring.get("risk_score", 0) < 60:
            if is_negligible:
                is_negligible = False
                notes.append("Micro-action approved: low risk single vehicle transfer allowed despite utility threshold.")

        if action.get("action") != "redirect":
            crowded = min(state, key=lambda z: state[z].get("free_slots", 0)) if state else "Unknown"
            free_slots = state.get(crowded, {}).get("free_slots", 0) if state else 0
            notes.append(f"Monitoring Phase: {crowded} currently has {free_slots} free slots. System is in holding state while levels are stable.")
            revised_action = {"action": "none"}
        elif is_negligible:
            # notes.append("Critic Veto: Action offers negligible network utility.") # Already added in _score_action
            revised_action = {"action": "none"}
        elif scoring["invalid_path"]:
            notes.append("Safety Alert: Redirect path is invalid. Protocol requires valid source and destination zones.")
            revised_action = {"action": "none"}
        elif safe_transfer <= 0:
            notes.append("Safety Alert: Destination zone is at capacity. Redirect blocked to prevent local overflow.")
            revised_action = {"action": "none"}
        else:
            requested = int(action.get("vehicles", 0))
            if safe_transfer < requested:
                notes.append(
                    f"Operational Adjustment: Transfer reduced from {requested} to {safe_transfer} vehicles to ensure destination stability."
                )
                revised_action["vehicles"] = safe_transfer
            revised_action["confidence"] = round(
                max(0.55, min(0.95, 1.0 - scoring["risk_probability"] + insight.get("confidence", 0.7) * 0.2)),
                2,
            )

        notes.extend(scoring["notes"])
        
        # ── Approval Logic: Softened Thresholds & Partial Approval ──
        # Risk levels: Low (<30), Medium (30-65), High (65-85), Critical (>85)
        risk_score = scoring["risk_score"]
        
        approved = False
        if revised_action.get("action") == "redirect" and not is_negligible:
            if risk_score < 80:
                approved = True
                notes.insert(0, f"Action Verified: The proposed redirect from {action.get('from')} to {action.get('to')} is tactically sound and supports the '{plan.get('strategy', 'allocation')}' mission.")
            elif risk_score < 95:
                approved = True
                partial_vehicles = max(1, int(revised_action.get("vehicles", 1)) // 2)
                revised_action["vehicles"] = partial_vehicles
                notes.insert(0, f"Critic Mitigation: High risk detected ({risk_score}), mitigating by reducing transfer volume to {partial_vehicles}.")
        
        if not approved and revised_action.get("action") == "redirect":
            notes.insert(0, f"Safety Override: Critical risk ({risk_score}). Reverting to system baseline.")
            revised_action = {"action": "none"}

        alternative_actions = self._suggest_alternatives(
            action,
            state,
            demand,
            event_context,
            operational_signals,
        )
        learning_feedback = self._build_learning_feedback(action, scoring, learning_profile)

        return {
            "approved": approved,
            "risk_level": self._risk_level(scoring["risk_score"]),
            "risk_score": scoring["risk_score"],
            "risk_probability": scoring["risk_probability"],
            "risk_factors": scoring["risk_factors"],
            "tool_calls": tool_calls,
            "tool_observations": {
                "learning_profile": learning_profile,
                "event_context": event_context,
                "operational_signals": operational_signals,
                "safe_transfer": safe_transfer,
            },
            "critic_notes": notes,
            "revised_action": revised_action if approved else {"action": "none"},
            "alternative_actions": alternative_actions,
            "replan_recommendation": self._build_replan_recommendation(scoring, alternative_actions),
            "learning_feedback": learning_feedback,
            "deterministic_review": True,
            "llm_advisory_used": False,
        }

    def _score_action(
        self,
        action,
        state,
        demand,
        insight,
        tools,
        event_context,
        operational_signals,
        learning_profile,
        tool_calls,
    ):
        notes = []
        risk_factors = {}
        from_zone = action.get("from")
        to_zone = action.get("to")
        requested = max(0, int(action.get("vehicles", 0) or 0))
        invalid_path = from_zone not in state or to_zone not in state or from_zone == to_zone
        safe_transfer = 0

        if action.get("action") != "redirect":
            risk_score = 12
            return {
                "risk_score": risk_score,
                "risk_probability": self._score_to_probability(risk_score),
                "risk_factors": {"no_redirect": 12},
                "invalid_path": False,
                "observations": {"safe_transfer": 0},
                "notes": notes,
            }

        if invalid_path:
            risk_factors["invalid_path"] = 70
            risk_score = 95
            return {
                "risk_score": risk_score,
                "risk_probability": self._score_to_probability(risk_score),
                "risk_factors": risk_factors,
                "invalid_path": True,
                "observations": {"safe_transfer": 0},
                "notes": notes,
            }

        safe_transfer = self._call_tool(
            tools,
            "estimate_transfer_capacity",
            from_zone,
            to_zone,
            requested,
            default=0,
            tool_calls=tool_calls,
        )
        capacity_ratio = safe_transfer / max(1, requested)
        risk_factors["capacity_risk"] = round((1 - capacity_ratio) * 30, 2)

        source_pressure = demand.get(from_zone, 0)
        source_free = state[from_zone]["free_slots"]
        total_hotspots = sum(1 for z, d in state.items() if d["free_slots"] <= 10)
        
        # Effectiveness Check: Is this redirect actually helpful?
        if total_hotspots == 0 and source_free > 15:
            risk_factors["low_utility_penalty"] = 45
            notes.append("Operational Warning: The system is currently stable. This redirect offers negligible network utility and may introduce unnecessary churn.")
        elif source_pressure < 4 and source_free > 12:
            risk_factors["weak_source_pressure"] = 20
            notes.append("Operational Note: Source pressure is too low to justify a proactive redirect at this scale.")

        destination_free_ratio = state[to_zone]["free_slots"] / max(1, state[to_zone]["total_slots"])
        if destination_free_ratio < 0.12:
            risk_factors["destination_congestion"] = 22
            notes.append("Destination is already close to capacity.")

        if event_context.get("severity") in {"high", "critical"} and from_zone != event_context.get("focus_zone"):
            risk_factors["event_misalignment"] = 20
            notes.append("High-severity event flow should prioritize the official focus zone.")

        if operational_signals.get("blocked_zone") in {from_zone, to_zone}:
            risk_factors["blocked_zone"] = 25
            notes.append("A blocked zone affects the proposed route.")
        queue_length = operational_signals.get("queue_length", 0)
        if queue_length >= 4:
            risk_factors["queue_pressure"] = min(15, queue_length * 2)
            notes.append("Queue pressure is elevated, so the critic is applying stricter routing checks.")

        route_reward = learning_profile.get("route_profile", {}).get("avg_reward", 0.0)
        if route_reward < -2.5:
            risk_factors["poor_route_history"] = 24
            notes.append("Historical route performance is poor for this transfer.")
        elif route_reward > 2:
            risk_factors["positive_route_history"] = -8

        posterior_risk = insight.get("posteriors", {}).get(to_zone)
        if posterior_risk is None:
            posterior_risk = insight.get("normalized_posteriors", {}).get(to_zone, 0.25)
        uncertainty = insight.get("uncertainty", {}).get("entropy", 0.0)
        risk_factors["destination_posterior_risk"] = round(float(posterior_risk) * 22, 2)
        risk_factors["uncertainty_penalty"] = round(min(15, float(uncertainty) * 3), 2)

        time_penalty = self._time_penalty(event_context)
        if time_penalty:
            risk_factors["time_window_pressure"] = time_penalty

        risk_score = round(max(0, min(100, sum(risk_factors.values()))), 2)
        return {
            "risk_score": risk_score,
            "risk_probability": self._score_to_probability(risk_score),
            "risk_factors": risk_factors,
            "invalid_path": False,
            "observations": {"safe_transfer": safe_transfer},
            "notes": notes,
        }

    def _ask_llm_for_review(
        self,
        deterministic_review,
        plan,
        state,
        demand,
        insight,
        event_context,
        operational_signals,
        learning_profile,
        scenario_mode,
    ):
        context = {
            "deterministic_review": deterministic_review,
            "plan": plan,
            "state": state,
            "demand": demand,
            "insight": insight,
            "event_context": event_context,
            "operational_signals": operational_signals,
            "learning_profile": learning_profile,
            "scenario_mode": scenario_mode,
        }
        schema_text = """
{
  "approved": true,
  "risk_level": "low",
  "critic_notes": ["note"],
  "revised_action": {
    "action": "redirect or none",
    "from": "zone",
    "to": "zone",
    "vehicles": 0,
    "reason": "string",
    "confidence": 0.0
  }
}
"""
        try:
            return ask_llm_for_structured_json(
                "CriticAgent",
                context,
                schema_text,
                deterministic_review,
                system_instruction=(
                    "You are an advisory critic. Do not override deterministic safety checks. "
                    "You may add notes or make a redirect stricter, but never approve a high-risk deterministic rejection."
                ),
                force=reasoning_budget.get("force_llm", False),
            )
        except Exception:
            return deterministic_review

    def _merge_llm_review(self, deterministic_review, llm_review):
        if not isinstance(llm_review, dict):
            deterministic_review["critic_notes"].append("LLM advisory review was ignored because it was not structured JSON.")
            return deterministic_review
        if llm_review.get("deterministic_review"):
            return deterministic_review
        if not self._is_valid_llm_review(llm_review):
            deterministic_review["critic_notes"].append("LLM advisory review was ignored because its structure was invalid.")
            return deterministic_review

        merged = deepcopy(deterministic_review)
        llm_notes = [
            note for note in llm_review.get("critic_notes", [])
            if isinstance(note, str) and note not in merged["critic_notes"]
        ]
        merged["critic_notes"].extend(llm_notes[:3])

        if deterministic_review["risk_score"] >= 70:
            merged["llm_advisory_used"] = bool(llm_notes)
            merged["critic_notes"].append("Deterministic safety gate retained control over high-risk action.")
            return merged

        llm_action = llm_review.get("revised_action", {})
        if llm_action.get("action") == "none" and merged.get("approved"):
            merged["approved"] = False
            merged["risk_level"] = max(merged["risk_level"], "medium", key=self._risk_rank)
            merged["revised_action"] = {"action": "none"}
            merged["critic_notes"].append("LLM advisory downgraded the action to no-op.")
        elif self._is_safe_stricter_action(merged.get("revised_action", {}), llm_action):
            merged["revised_action"] = llm_action
            merged["critic_notes"].append("LLM advisory reduced the action within deterministic safety limits.")

        merged["llm_advisory_used"] = True
        return merged

    def _is_valid_llm_review(self, review):
        return (
            isinstance(review, dict)
            and isinstance(review.get("critic_notes", []), list)
            and isinstance(review.get("revised_action", {}), dict)
            and review.get("risk_level", "low") in {"low", "medium", "high"}
        )

    def _is_safe_stricter_action(self, deterministic_action, llm_action):
        if deterministic_action.get("action") != "redirect" or llm_action.get("action") != "redirect":
            return False
        return (
            deterministic_action.get("from") == llm_action.get("from")
            and deterministic_action.get("to") == llm_action.get("to")
            and 0 < int(llm_action.get("vehicles", 0)) <= int(deterministic_action.get("vehicles", 0))
        )

    def _suggest_alternatives(self, action, state, demand, event_context, operational_signals):
        if not state:
            return []
        from_zone = action.get("from") if action.get("from") in state else max(
            state,
            key=lambda zone: demand.get(zone, 0),
        )
        blocked_zone = operational_signals.get("blocked_zone")
        candidates = [
            zone for zone in state
            if zone != from_zone and zone != blocked_zone
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
            vehicles = max(1, min(8, state[zone]["free_slots"], int(demand.get(from_zone, 0) / 2) or 1))
            alternatives.append(
                {
                    "action": "redirect",
                    "from": from_zone,
                    "to": zone,
                    "vehicles": vehicles,
                    "reason": f"{zone} is a safer candidate with {state[zone]['free_slots']} free slots.",
                }
            )
        return alternatives

    def _build_replan_recommendation(self, scoring, alternatives):
        if scoring["risk_score"] >= 70:
            return {
                "required": True,
                "reason": "Risk score is high; planner should re-evaluate route and vehicle count.",
                "suggested_action": alternatives[0] if alternatives else {"action": "none"},
            }
        if scoring["risk_score"] >= 40:
            return {
                "required": False,
                "reason": "Medium risk; continue only with reduced capacity and monitoring.",
                "suggested_action": alternatives[0] if alternatives else {"action": "none"},
            }
        return {"required": False, "reason": "Risk is within acceptable bounds.", "suggested_action": None}

    def _build_learning_feedback(self, action, scoring, learning_profile):
        return {
            "route": f"{action.get('from')}->{action.get('to')}" if action.get("from") and action.get("to") else None,
            "critic_risk_score": scoring["risk_score"],
            "critic_risk_probability": scoring["risk_probability"],
            "recommended_bias_adjustment": -0.05 if scoring["risk_score"] >= 70 else 0.03,
            "prior_route_profile": learning_profile.get("route_profile", {}),
        }

    def _update_critic_memory(self, review):
        route = review.get("learning_feedback", {}).get("route")
        if not route:
            return
        profile = self.route_risk_memory.setdefault(route, {"reviews": 0, "avg_risk_score": 0.0})
        profile["reviews"] += 1
        count = profile["reviews"]
        profile["avg_risk_score"] = round(
            ((profile["avg_risk_score"] * (count - 1)) + review.get("risk_score", 0.0)) / count,
            2,
        )

    def _log_review(self, review):
        level = "ERROR" if review.get("risk_level") == "high" else "WARNING" if review.get("risk_level") == "medium" else "INFO"
        self.logger.log(
            review.get("tool_observations", {}).get("event_context", {}).get("step", "-"),
            "critic_review",
            {
                "approved": review.get("approved"),
                "risk_level": review.get("risk_level"),
                "risk_score": review.get("risk_score"),
                "risk_probability": review.get("risk_probability"),
                "notes": review.get("critic_notes", []),
            },
            level=level,
        )

    def _call_tool(self, tools, name, *args, default=None, keyword_args=None, tool_calls=None):
        keyword_args = keyword_args or {}
        if tool_calls is not None and name not in tool_calls:
            tool_calls.append(name)
        tool = tools.get(name) if isinstance(tools, dict) else None
        if not callable(tool):
            return default
        try:
            if keyword_args:
                return tool(*args, **keyword_args)
            return tool(*args)
        except Exception:
            return default

    def _risk_level(self, risk_score):
        if risk_score >= 70:
            return "high"
        if risk_score >= 40:
            return "medium"
        return "low"

    def _risk_rank(self, level):
        return {"low": 0, "medium": 1, "high": 2}.get(level, 0)

    def _score_to_probability(self, risk_score):
        return round(max(0.01, min(0.99, risk_score / 100)), 3)

    def _time_penalty(self, event_context):
        window = event_context.get("time_window", "")
        if window.startswith("08") or window.startswith("09") or window.startswith("17") or window.startswith("18"):
            return 5
        return 0
