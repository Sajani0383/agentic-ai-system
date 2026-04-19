import json
import logging

from llm.client import get_llm, get_llm_status
from llm.prompts import PLANNER_PROMPT_TEMPLATE, STRUCTURED_COPILOT_TEMPLATE, CHAT_AGENT_TEMPLATE
from llm.parser import extract_json_payload, validate_action_schema
from llm.fallback import summarize_state, build_advanced_fallback
from llm.chat import get_local_chat_response

# Preserved solely for backward-compatible interface contracts
LAST_LLM_ERROR = ""

def build_fallback_action(state):
    return build_advanced_fallback(state)

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
        decision = extract_json_payload(response)
        if not validate_action_schema(decision):
            return fallback
        return decision
    except Exception as e:
        logging.error(f"Fallback trigged due to exception in json reasoning sequence: {e}")
        return fallback

def _compact_context(context):
    """Prunes large context dictionaries to ensure reliability on constrained budgets."""
    if not isinstance(context, dict):
        return context
    
    compact = {}
    prune_keys = {"recent_cycles", "recent_states", "history", "tool_observations", "tool_calls"}
    
    for k, v in context.items():
        if k in prune_keys:
            if isinstance(v, list):
                compact[k] = v[-2:] if v else [] # Only keep last 2 items
            else:
                continue
        elif isinstance(v, dict) and len(str(v)) > 2000:
            # Recursively compact or truncate
            compact[k] = {sk: sv for sk, sv in v.items() if len(str(sv)) < 500}
        else:
            compact[k] = v
            
    return compact

def ask_llm_for_structured_json(agent_name, context, schema_text, fallback, system_instruction=None, force=False):
    llm = get_llm(force=force)
    if llm is None:
        return fallback

    # 1. Attempt with standard context
    # 2. If it fails or is too large, attempt with compact context
    
    contexts_to_try = [context, _compact_context(context)]
    
    for i, ctx in enumerate(contexts_to_try):
        prompt = STRUCTURED_COPILOT_TEMPLATE.format(
            agent_name=agent_name,
            system_instruction=system_instruction or "Use the provided context carefully and return valid JSON only.",
            context=json.dumps(ctx, indent=2),
            schema_text=schema_text
        )

        try:
            response = llm.invoke(prompt).content.strip()
            payload = extract_json_payload(response)
            if isinstance(payload, dict):
                return payload
        except Exception as e:
            logging.warning(f"LLM attempt {i+1} for {agent_name} failed: {e}")
            if i == 0: continue # Try compact
            
    return fallback

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
