from llm_reasoning import ask_llm_for_structured_json


class CriticAgent:
    def review(self, plan, state, demand, insight, tools):
        proposed_action = plan.get("proposed_action", {"action": "none"})
        tool_calls = list(plan.get("tool_calls", []))
        tool_calls.append("estimate_transfer_capacity")
        event_context = tools["get_event_context"]()
        tool_calls.append("get_event_context")
        scenario_mode = tools["get_scenario_mode"]()
        tool_calls.append("get_scenario_mode")
        learning_profile = tools["get_learning_profile"](
            scenario_mode=scenario_mode,
            from_zone=proposed_action.get("from"),
            to_zone=proposed_action.get("to"),
        )
        tool_calls.append("get_learning_profile")

        fallback_review = self._build_fallback_review(
            proposed_action,
            state,
            demand,
            insight,
            plan,
            tools,
            event_context,
            learning_profile,
        )
        context = {
            "plan": plan,
            "state": state,
            "demand": demand,
            "insight": insight,
            "event_context": event_context,
            "learning_profile": learning_profile,
            "scenario_mode": scenario_mode,
        }
        schema_text = """
{
  "approved": true,
  "risk_level": "low",
  "tool_calls": ["tool_name"],
  "tool_observations": {
    "learning_profile": {},
    "event_context": {}
  },
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
        review = ask_llm_for_structured_json(
            "CriticAgent",
            context,
            schema_text,
            fallback_review,
            system_instruction=(
                "You are the default safety and quality critic. Approve only if the action is safe, "
                "aligned with the event focus zone, and consistent with learning_profile."
            ),
        )
        review.setdefault("tool_calls", tool_calls)
        review.setdefault(
            "tool_observations",
            {"learning_profile": learning_profile, "event_context": event_context},
        )
        return review

    def _build_fallback_review(self, action, state, demand, insight, plan, tools, event_context, learning_profile):
        notes = []
        revised_action = dict(action)
        if action.get("action") != "redirect":
            notes.append("No redirect required because the planner did not propose a transfer.")
            return {
                "approved": True,
                "risk_level": "low",
                "tool_calls": plan.get("tool_calls", []),
                "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
                "critic_notes": notes,
                "revised_action": {"action": "none"},
            }

        from_zone = action.get("from")
        to_zone = action.get("to")
        requested = int(action.get("vehicles", 0))
        safe_transfer = tools["estimate_transfer_capacity"](from_zone, to_zone, requested)

        if from_zone not in state or to_zone not in state or from_zone == to_zone:
            notes.append("Redirect path is invalid, so execution is blocked.")
            return {
                "approved": False,
                "risk_level": "high",
                "tool_calls": plan.get("tool_calls", []),
                "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
                "critic_notes": notes,
                "revised_action": {"action": "none"},
            }

        if safe_transfer <= 0:
            notes.append("Destination has no safe capacity for the requested transfer.")
            return {
                "approved": False,
                "risk_level": "high",
                "tool_calls": plan.get("tool_calls", []),
                "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
                "critic_notes": notes,
                "revised_action": {"action": "none"},
            }

        if safe_transfer < requested:
            notes.append(
                f"Transfer reduced from {requested} to {safe_transfer} to stay within safe capacity limits."
            )
            revised_action["vehicles"] = safe_transfer

        if event_context.get("severity") in {"high", "critical"} and from_zone != event_context.get("focus_zone"):
            notes.append("Critical event flow should prioritize the official focus zone before other moves.")
            return {
                "approved": False,
                "risk_level": "medium",
                "tool_calls": plan.get("tool_calls", []),
                "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
                "critic_notes": notes,
                "revised_action": {"action": "none"},
            }

        if demand.get(from_zone, 0) < 5 and state[from_zone]["free_slots"] > 12:
            notes.append("Source pressure is modest, so the redirect is downgraded to no-op.")
            return {
                "approved": False,
                "risk_level": "medium",
                "tool_calls": plan.get("tool_calls", []),
                "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
                "critic_notes": notes,
                "revised_action": {"action": "none"},
            }

        route_reward = learning_profile.get("route_profile", {}).get("avg_reward", 0.0)
        if route_reward < -2.5:
            notes.append("Historical route performance is poor, so the move is downgraded for this cycle.")
            return {
                "approved": False,
                "risk_level": "medium",
                "tool_calls": plan.get("tool_calls", []),
                "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
                "critic_notes": notes,
                "revised_action": {"action": "none"},
            }

        revised_action["confidence"] = round(max(0.7, insight.get("confidence", 0.75)), 2)
        notes.append(
            f"Redirect is safe and supports the active {plan.get('strategy', 'allocation')} strategy."
        )
        return {
            "approved": True,
            "risk_level": "low",
            "tool_calls": plan.get("tool_calls", []),
            "tool_observations": {"learning_profile": learning_profile, "event_context": event_context},
            "critic_notes": notes,
            "revised_action": revised_action,
        }
