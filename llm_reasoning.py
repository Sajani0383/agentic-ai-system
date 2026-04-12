import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
elif ENV_EXAMPLE_PATH.exists():
    load_dotenv(dotenv_path=ENV_EXAMPLE_PATH, override=True)


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LAST_LLM_ERROR = ""


def _set_last_llm_error(message):
    global LAST_LLM_ERROR
    LAST_LLM_ERROR = message


def _is_terminal_llm_error(message):
    text = (message or "").upper()
    return any(
        marker in text
        for marker in [
            "API_KEY_INVALID",
            "API KEY EXPIRED",
            "INVALID_ARGUMENT",
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
        ]
    )


def _is_connectivity_llm_error(message):
    text = (message or "").upper()
    return any(
        marker in text
        for marker in [
            "CONNECTERROR",
            "CONNECTIONERROR",
            "NODENAME NOR SERVNAME PROVIDED",
            "TEMPORARY FAILURE IN NAME RESOLUTION",
            "TLS",
            "SSL",
            "CERTIFICATE",
            "TIMEOUT",
        ]
    )


def _get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    cleaned = api_key.strip()
    if api_key.strip().lower() in {
        "your_google_api_key_here",
        "your_real_key_here",
        "replace_me",
    }:
        return None
    # Most Gemini API keys start with AIza; reject obviously malformed placeholders early.
    if len(cleaned) < 20 or not cleaned.startswith("AIza"):
        return None
    return cleaned


def summarize_state(state):
    crowded = min(state, key=lambda zone: state[zone]["free_slots"])
    best = max(state, key=lambda zone: state[zone]["free_slots"])
    return {
        "most_crowded": crowded,
        "best_zone": best,
        "crowded_free_slots": state[crowded]["free_slots"],
        "best_free_slots": state[best]["free_slots"],
    }


def _build_fallback_action(state):
    summary = summarize_state(state)
    crowded = summary["most_crowded"]
    best = summary["best_zone"]

    if state[crowded]["free_slots"] <= 10 and crowded != best:
        vehicles = max(1, min(8, (12 - state[crowded]["free_slots"])))
        return {
            "action": "redirect",
            "from": crowded,
            "to": best,
            "vehicles": vehicles,
            "reason": f"{crowded} is congested and {best} has more free slots.",
            "confidence": 0.7,
        }

    return {
        "action": "none",
        "from": crowded,
        "to": best,
        "vehicles": 0,
        "reason": "Current balance is acceptable. No redirect needed.",
        "confidence": 0.65,
    }


class GeminiLLMWrapper:
    def __init__(self, api_key, model):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def invoke(self, prompt):
        last_exc = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt)
                text = getattr(response, "text", None)
                if not text:
                    text = ""
                    for candidate in getattr(response, "candidates", []) or []:
                        content = getattr(candidate, "content", None)
                        parts = getattr(content, "parts", []) if content else []
                        for part in parts:
                            part_text = getattr(part, "text", "")
                            if part_text:
                                text += part_text
                _set_last_llm_error("")
                return SimpleNamespace(content=(text or "").strip())
            except Exception as exc:
                last_exc = exc
                message = f"{type(exc).__name__}: {exc}"
                is_retryable = "503" in message or "UNAVAILABLE" in message.upper()
                if is_retryable and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                _set_last_llm_error(message)
                raise
        if last_exc:
            raise last_exc


def get_llm_status():
    enable_llm = os.getenv("ENABLE_LLM", "true").lower() in {"1", "true", "yes"}
    api_key = _get_api_key()
    status = {
        "enabled": enable_llm,
        "api_key_present": bool(api_key),
        "env_file_present": ENV_PATH.exists(),
        "env_path": str(ENV_PATH),
        "model": DEFAULT_GEMINI_MODEL,
        "provider": "google-genai",
        "available": False,
        "message": "",
        "last_error": LAST_LLM_ERROR,
    }

    if not enable_llm:
        status["message"] = "LLM mode is disabled by ENABLE_LLM."
        return status

    if not api_key:
        status["message"] = "No valid GOOGLE_API_KEY or GEMINI_API_KEY found in .env."
        return status

    try:
        from google import genai  # noqa: F401
    except Exception as exc:
        status["message"] = f"google-genai import failed: {type(exc).__name__}"
        return status

    status["available"] = True
    status["message"] = "Gemini SDK is configured. Live API validation happens on first request."
    if LAST_LLM_ERROR:
        if _is_terminal_llm_error(LAST_LLM_ERROR):
            status["available"] = False
            status["message"] = (
                "Gemini is disabled because the configured API key is invalid or expired. "
                "Update GOOGLE_API_KEY in .env to re-enable live Gemini reasoning."
            )
            return status
        if "503" in LAST_LLM_ERROR or "UNAVAILABLE" in LAST_LLM_ERROR.upper():
            status["message"] = (
                "Gemini is configured correctly, but the provider is temporarily overloaded. "
                "The dashboard will retry and use local fallback until Gemini responds again."
            )
        elif _is_connectivity_llm_error(LAST_LLM_ERROR):
            status["message"] = (
                "Gemini is configured, but this runtime could not reach the Gemini API. "
                "Check DNS, outbound network access, and local TLS/SSL support."
            )
        else:
            status["message"] = f"Gemini configured, but last call failed: {LAST_LLM_ERROR}"
    return status


def get_llm():
    status = get_llm_status()
    if not status["available"]:
        return None

    api_key = _get_api_key()
    try:
        return GeminiLLMWrapper(api_key=api_key, model=DEFAULT_GEMINI_MODEL)
    except Exception:
        return None


def _extract_json_payload(text):
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = next((part for part in parts if "{" in part and "}" in part), cleaned)
    cleaned = cleaned.replace("json", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= 0:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start:end])


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


def ask_llm_for_json_decision(state, demand, insight, memory_metrics):
    fallback = _build_fallback_action(state)
    llm = get_llm()

    if llm is None:
        return fallback

    prompt = f"""
You are an intelligent parking control agent.

Parking state:
{json.dumps(state, indent=2)}

Demand pressure:
{json.dumps(demand, indent=2)}

Inference:
{json.dumps(insight, indent=2)}

Memory metrics:
{json.dumps(memory_metrics, indent=2)}

Return JSON only in this schema:
{{
  "action": "redirect" or "none",
  "from": "zone name",
  "to": "zone name",
  "vehicles": 0,
  "reason": "short reason",
  "confidence": 0.0
}}
"""

    try:
        response = llm.invoke(prompt).content.strip()
        decision = _extract_json_payload(response)
        if "action" not in decision:
            return fallback
        return decision
    except Exception:
        return fallback


def ask_llm_for_structured_json(agent_name, context, schema_text, fallback, system_instruction=None):
    llm = get_llm()
    if llm is None:
        return fallback

    prompt = f"""
You are the {agent_name} for an autonomous parking orchestration system.

{system_instruction or "Use the provided context carefully and return valid JSON only."}

Context:
{json.dumps(context, indent=2)}

Return JSON only in this schema:
{schema_text}
"""

    try:
        response = llm.invoke(prompt).content.strip()
        payload = _extract_json_payload(response)
        return payload if isinstance(payload, dict) else fallback
    except Exception:
        return fallback


def get_operational_briefing(state, latest_result, event_context, learning_profile):
    summary = summarize_state(state)
    kpis = latest_result.get("kpis", {})
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
    llm = get_llm()
    if llm is None:
        return fallback

    schema_text = """
{
  "headline": "string",
  "narrative": "string",
  "prediction": "string",
  "suggestions": ["string"],
  "decision_commentary": "string"
}
"""
    context = {
        "state": state,
        "latest_result": latest_result,
        "event_context": event_context,
        "learning_profile": learning_profile,
    }
    return ask_llm_for_structured_json(
        "OperationsCopilot",
        context,
        schema_text,
        fallback,
        system_instruction=(
            "You are a proactive parking copilot. Summarize current operations in short, plain language, "
            "comment on the latest agent decision, make one near-term prediction, and give three specific suggestions."
        ),
    )


def _format_zone_table_lines(state, fields):
    lines = []
    for zone, data in state.items():
        values = [f"{field}: {data.get(field, 0)}" for field in fields]
        lines.append(f"- {zone}: " + ", ".join(values))
    return "\n".join(lines)


def get_local_chat_response(state, query):
    summary = summarize_state(state)
    query_lower = query.lower()

    if (
        "occupied in each" in query_lower
        or "occupied slots" in query_lower
        or "all the slots occupied" in query_lower
        or "zone by zone" in query_lower
    ):
        return "Occupied slots by zone:\n" + _format_zone_table_lines(state, ["occupied", "free_slots", "total_slots"])

    if "free slots in each" in query_lower or "free slots" in query_lower and "each" in query_lower:
        return "Free slots by zone:\n" + _format_zone_table_lines(state, ["free_slots", "occupied", "total_slots"])

    if "entries and exits" in query_lower or "vehicle movement" in query_lower:
        return "Current entries and exits by zone:\n" + _format_zone_table_lines(state, ["entry", "exit", "free_slots"])

    if "which zone is full" in query_lower or "fully occupied" in query_lower:
        fullest = max(state, key=lambda zone: state[zone]["occupied"] / max(1, state[zone]["total_slots"]))
        fullness = round((state[fullest]["occupied"] / max(1, state[fullest]["total_slots"])) * 100, 1)
        return (
            f"{fullest} is the closest to full right now at {fullness}% occupancy, "
            f"with {state[fullest]['free_slots']} free slots remaining."
        )

    if "current event" in query_lower or "what event" in query_lower:
        return "The current event context is available in the runtime snapshot. Ask for the event plus the latest allocation to get a full operational update."

    if "slow" in query_lower or "filling" in query_lower:
        slowest = min(
            state,
            key=lambda zone: state[zone]["entry"] - state[zone]["exit"],
        )
        return (
            f"{slowest} is filling the slowest right now. "
            f"It has entry {state[slowest]['entry']} and exit {state[slowest]['exit']}, "
            f"with {state[slowest]['free_slots']} free slots remaining."
        )

    if "best" in query_lower or "available" in query_lower:
        best = summary["best_zone"]
        return (
            f"{best} is the best current choice with "
            f"{state[best]['free_slots']} free slots."
        )

    if "crowd" in query_lower or "full" in query_lower or "congestion" in query_lower:
        crowded = summary["most_crowded"]
        return (
            f"{crowded} is the most congested zone right now with "
            f"{state[crowded]['free_slots']} free slots."
        )

    return (
        f"Current system summary: {summary['most_crowded']} is most crowded, "
        f"{summary['best_zone']} is best available, and the suggested action is to "
        f"redirect vehicles only when congestion remains high."
    )


class ParkingLLMAgent:
    def __init__(self, tools):
        self.tool_map = {tool.name.lower(): tool for tool in tools}

    def _run_tool(self, name):
        tool = self.tool_map.get(name)
        if tool is None:
            return ""
        try:
            return tool.func()
        except TypeError:
            return tool.func(None)

    def _build_fallback_output(self, query):
        state_text = self._run_tool("get state")
        decision_text = self._run_tool("decision")
        parts = [query]
        if decision_text:
            parts.append(decision_text)
        if state_text:
            parts.append(f"State Snapshot:\n{state_text}")
        return "\n\n".join(parts)

    def invoke(self, query):
        llm = get_llm()
        if llm is None:
            return {"output": self._build_fallback_output(query)}

        tool_outputs = {
            "state": self._run_tool("get state"),
            "decision": self._run_tool("decision"),
            "predict_demand": self._run_tool("predict demand"),
            "trend": self._run_tool("trend"),
            "metrics": self._run_tool("metrics"),
        }
        prompt = f"""
You are an intelligent parking operations assistant.

User request:
{query}

Available tool outputs:
{json.dumps(tool_outputs, indent=2)}

Write a short operational answer that:
1. directly answers the user request,
2. identifies the best zone and the most congested zone when possible,
3. uses the tool outputs only, and
4. does not mention missing tools or internal implementation details.
"""
        try:
            response = llm.invoke(prompt).content.strip()
            if response:
                return {"output": response}
        except Exception:
            pass
        return {"output": self._build_fallback_output(query)}

    def run(self, query):
        return self.invoke(query)["output"]


def create_llm_agent(tools):
    return ParkingLLMAgent(tools)
