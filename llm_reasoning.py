import json
import logging
from copy import deepcopy

from llm.client import (
    ENV_EXAMPLE_PATH,
    ENV_PATH,
    _is_connectivity_llm_error,
    _is_quota_llm_error,
    get_llm,
    get_llm_status,
)
from llm.prompts import PLANNER_PROMPT_TEMPLATE, STRUCTURED_COPILOT_TEMPLATE, CHAT_AGENT_TEMPLATE
from llm.parser import extract_json_payload, normalize_action_schema, validate_action_schema
from llm.fallback import summarize_state, build_advanced_fallback
from llm.chat import get_local_chat_response

# Preserved solely for backward-compatible interface contracts
LAST_LLM_ERROR = ""

def _with_llm_runtime(fallback, **runtime):
    payload = deepcopy(fallback) if isinstance(fallback, dict) else fallback
    if isinstance(payload, dict):
        payload["_llm_runtime"] = runtime
    return payload

def build_fallback_action(state):
    return build_advanced_fallback(state)

_build_fallback_action = build_fallback_action

def get_operational_reasoning(state):
    summary = summarize_state(state)
    crowded = summary["most_crowded"]
    best = summary["best_zone"]
    return {
        "source": "local_operational_summary",
        "text": (
            f"Most crowded zone: {crowded}. "
            f"Best available zone: {best}. "
            f"Suggested action: redirect incoming arrivals from {crowded} to {best} if congestion persists."
        ),
    }

def get_llm_decision(state):
    return get_operational_reasoning(state)["text"]

def get_ai_reasoning(zone, free_slots, level):
    if free_slots <= 5:
        return f"{zone} is highly congested with {free_slots} free slots. Priority should be redirection."
    if level.upper() == "HIGH":
        return f"{zone} has elevated congestion risk. Monitor closely and prefer overflow routing."
    return f"{zone} is stable. Maintain current flow and continue monitoring."

def ask_llm_for_json_decision(state, demand, insight, memory_metrics, force=False):
    fallback = build_advanced_fallback(state, demand=demand, insight=insight)
    llm = get_llm(force=force)

    if llm is None:
        return fallback

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        state=json.dumps(state, indent=2),
        demand=json.dumps(demand, indent=2),
        insight=json.dumps(insight, indent=2),
        memory_metrics=json.dumps(memory_metrics, indent=2)
    )

    try:
        response = llm.invoke(prompt).content.strip()
        decision = normalize_action_schema(extract_json_payload(response))
        if not validate_action_schema(decision):
            return fallback
        return decision
    except Exception as e:
        logging.warning(f"Fallback trigged due to exception in json reasoning sequence: {e}")
        return fallback

def _compact_context(context):
    """Prunes large context dictionaries to ensure reliability on constrained budgets."""
    if not isinstance(context, dict):
        return context

    state = context.get("state", {})
    demand = context.get("demand", {})
    insight = context.get("insight", {})
    event_context = context.get("event_context", {})
    operational_signals = context.get("operational_signals", {})
    learning_profile = context.get("learning_profile", {})
    recent_cycles = context.get("recent_cycles", [])

    top_blocks = []
    if isinstance(state, dict) and state:
        ranked = sorted(
            state.items(),
            key=lambda item: (
                item[1].get("free_slots", 0),
                -item[1].get("occupied", 0),
            ),
        )
        for zone, zone_state in ranked[:3]:
            top_blocks.append(
                {
                    "name": zone,
                    "free_slots": int(zone_state.get("free_slots", 0) or 0),
                    "occupied": int(zone_state.get("occupied", 0) or 0),
                    "capacity": int(
                        zone_state.get("total_slots", zone_state.get("capacity", 0)) or 0
                    ),
                    "demand": int(demand.get(zone, 0) or 0),
                }
            )

    total_free_slots = sum(
        int(zone_state.get("free_slots", 0) or 0)
        for zone_state in state.values()
    ) if isinstance(state, dict) else 0

    compact = {
        "top_blocks": top_blocks,
        "queue": int(operational_signals.get("queue_length", 0) or 0),
        "entropy": round(float(insight.get("uncertainty", {}).get("entropy", 0.0) or 0.0), 3),
        "free_slots": total_free_slots,
        "event": {
            "name": event_context.get("name", "Normal Flow"),
            "severity": event_context.get("severity", "low"),
            "focus_zone": event_context.get("focus_zone", ""),
        },
        "learning": {
            "recent_reward_avg": round(float(learning_profile.get("recent_reward_avg", 0.0) or 0.0), 3),
            "blocked_routes": list((learning_profile.get("blocked_routes") or [])[:4]),
            "latest_insight": learning_profile.get("latest_learning_insight", ""),
        },
        "recent_outcome": {},
    }
    if isinstance(recent_cycles, list) and recent_cycles:
        last_cycle = recent_cycles[-1] if isinstance(recent_cycles[-1], dict) else {}
        compact["recent_outcome"] = {
            "reward": round(float(last_cycle.get("reward", {}).get("agentic_reward_score", 0.0) or 0.0), 3),
            "action": last_cycle.get("execution_output", {}).get("final_action", {}).get("action", "none"),
        }
    return compact

def _normalize_structured_payload(payload, fallback):
    if not isinstance(payload, dict):
        return payload
    normalized = deepcopy(payload)
    fallback_action = deepcopy(fallback.get("proposed_action", {})) if isinstance(fallback, dict) else {}

    if "proposed_action" not in normalized and any(key in normalized for key in ("action", "from", "to", "vehicles")):
        normalized["proposed_action"] = {
            "action": normalized.get("action"),
            "from": normalized.get("from"),
            "to": normalized.get("to"),
            "vehicles": normalized.get("vehicles"),
            "reason": normalized.get("reason", ""),
            "confidence": normalized.get("confidence", fallback_action.get("confidence", 0.0)),
        }

    if isinstance(normalized.get("proposed_action"), dict):
        normalized["proposed_action"] = normalize_action_schema(normalized["proposed_action"])
        normalized["proposed_action"].setdefault("reason", normalized.get("rationale", "") or fallback_action.get("reason", ""))
        normalized["proposed_action"].setdefault("confidence", fallback_action.get("confidence", 0.0))

    alt = normalized.get("alternative_actions")
    if isinstance(alt, dict):
        normalized["alternative_actions"] = [normalize_action_schema(alt)]
    elif isinstance(alt, list):
        normalized["alternative_actions"] = [
            normalize_action_schema(item) if isinstance(item, dict) else item
            for item in alt
        ]
    return normalized

def ask_llm_for_structured_json(agent_name, context, schema_text, fallback, system_instruction=None, force=False):
    llm = get_llm(force=force)
    if llm is None:
        status = get_llm_status(ignore_backoff=force)
        return _with_llm_runtime(
            fallback,
            requested=True,
            used=False,
            fallback_used=True,
            fallback_reason=status.get("message", "Gemini unavailable; local fallback used."),
            error=status.get("last_error", ""),
            source="local_fallback",
        )

    compact_context = _compact_context(context)
    prompt = STRUCTURED_COPILOT_TEMPLATE.format(
        agent_name=agent_name,
        system_instruction=system_instruction or "Use the provided context carefully and return valid JSON only.",
        context=json.dumps(compact_context, indent=2),
        schema_text=schema_text
    )

    try:
        response = llm.invoke(prompt).content.strip()
        payload = _normalize_structured_payload(extract_json_payload(response), fallback)
        if isinstance(payload, dict):
            payload["_llm_runtime"] = {
                "requested": True,
                "used": True,
                "fallback_used": False,
                "fallback_reason": "",
                "error": "",
                "source": "gemini",
            }
            return payload
    except Exception as e:
        logging.warning(f"LLM attempt 1 for {agent_name} failed: {e}")
        if 'response' in locals():
            logging.warning(f"{agent_name} raw LLM output snippet: {response[:400]}")
            
    status = get_llm_status(ignore_backoff=True)
    return _with_llm_runtime(
        fallback,
        requested=True,
        used=False,
        fallback_used=True,
        fallback_reason="Gemini request failed; deterministic/local fallback completed the decision.",
        error=status.get("last_error", ""),
        source="gemini_failed_fallback",
    )

def get_operational_briefing(state, latest_result, event_context, learning_profile, use_llm=True):
    summary = summarize_state(state)
    action = latest_result.get("action", {})
    critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
    operational_signals = latest_result.get("operational_signals", {})
    
    fallback = {
        "headline": f"{event_context.get('name', 'Campus')} parking is stable, with {summary['best_zone']} currently the best arrival zone.",
        "narrative": (
            f"The system is tracking {event_context.get('name', 'current conditions')} and currently "
            f"rates {summary['most_crowded']} as the busiest area while {summary['best_zone']} has the best free-space buffer."
        ),
        "prediction": (
            f"Next-step pressure is likely to build around {event_context.get('focus_zone', summary['most_crowded'])} "
            f"if queue length rises above {max(2, operational_signals.get('queue_length', 0) + 1)}."
        ),
        "suggestions": [
            f"Guide new arrivals toward {event_context.get('recommended_zone', summary['best_zone'])}.",
            f"Watch {event_context.get('focus_zone', summary['most_crowded'])} for congestion or denied-entry spikes.",
            "Use the benchmark tab to compare this scenario against the no-redirect baseline.",
        ],
        "decision_commentary": (
            action.get("reason")
            or (critic_notes[0] if critic_notes else "The agent is holding the current allocation because network pressure is manageable.")
        ),
    }

    schema_text = """
    {
      "headline": "string",
      "narrative": "string",
      "prediction": "string",
      "suggestions": ["string"],
      "decision_commentary": "string"
    }
    """
    
    context = {"state": state, "latest_result": latest_result, "event_context": event_context, "learning_profile": learning_profile}
    
    if not use_llm:
        return fallback

    return ask_llm_for_structured_json(
        "OperationsCopilot",
        context,
        schema_text,
        fallback,
        system_instruction="You are a proactive parking copilot. Summarize current operations in short, plain language, comment on the latest agent decision, make one near-term prediction, and give three specific suggestions."
    )

class ParkingLLMAgent:
    def __init__(self, tools):
        self.tool_map = {tool.name.lower(): tool for tool in tools}

    def _run_tool(self, name):
        tool = self.tool_map.get(name)
        if tool is None: return ""
        try: return tool.func()
        except TypeError: return tool.func(None)

    def _build_fallback_output(self, query):
        state_text = self._run_tool("get state")
        decision_text = self._run_tool("decision")
        parts = [query]
        if decision_text: parts.append(decision_text)
        if state_text: parts.append(f"State Snapshot:\n{state_text}")
        return "\n\n".join(parts)

    def invoke(self, query):
        llm = get_llm()
        if llm is None: return {"output": self._build_fallback_output(query)}

        tool_outputs = {
            "state": self._run_tool("get state"),
            "decision": self._run_tool("decision"),
            "predict_demand": self._run_tool("predict demand"),
            "trend": self._run_tool("trend"),
            "metrics": self._run_tool("metrics"),
        }
        
        prompt = CHAT_AGENT_TEMPLATE.format(
            query=query,
            tool_outputs=json.dumps(tool_outputs, indent=2)
        )
        
        try:
            response = llm.invoke(prompt).content.strip()
            if response: return {"output": response}
        except Exception as e:
            logging.error(f"Chat agent LLM hook failed. Reverting to structural blocks: {e}")
            
        return {"output": self._build_fallback_output(query)}

    def run(self, query):
        return self.invoke(query)["output"]


def create_llm_agent(tools):
    return ParkingLLMAgent(tools)
