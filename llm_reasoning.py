import json
import os
import time
from types import SimpleNamespace

from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env", override=True)
elif os.path.exists(".env.example"):
    load_dotenv(dotenv_path=".env.example", override=True)


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LAST_LLM_ERROR = ""


def _set_last_llm_error(message):
    global LAST_LLM_ERROR
    LAST_LLM_ERROR = message


def _get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    if api_key.strip().lower() in {
        "your_google_api_key_here",
        "your_real_key_here",
        "replace_me",
    }:
        return None
    return api_key


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
        "env_file_present": os.path.exists(".env"),
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
        status["message"] = "No valid GOOGLE_API_KEY or GEMINI_API_KEY found in .env or .env.example."
        return status

    try:
        from google import genai  # noqa: F401
    except Exception as exc:
        status["message"] = f"google-genai import failed: {type(exc).__name__}"
        return status

    status["available"] = True
    status["message"] = "Gemini SDK is configured. Live API validation happens on first request."
    if LAST_LLM_ERROR:
        if "503" in LAST_LLM_ERROR or "UNAVAILABLE" in LAST_LLM_ERROR.upper():
            status["message"] = (
                "Gemini is configured correctly, but the provider is temporarily overloaded. "
                "The dashboard will retry and use local fallback until Gemini responds again."
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


def get_llm_decision(state):
    summary = summarize_state(state)
    crowded = summary["most_crowded"]
    best = summary["best_zone"]
    return (
        f"Most crowded zone: {crowded}. "
        f"Best available zone: {best}. "
        f"Suggested action: redirect vehicles from {crowded} to {best} if congestion persists."
    )


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
        return "Ask the runtime event context from the dashboard cards or API to see the active campus event and routing strategy."

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

    def invoke(self, query):
        state_text = self.tool_map["get state"].func()
        decision_text = self.tool_map["decision"].func()
        return {"output": f"{query}\n\n{decision_text}\n\nState Snapshot:\n{state_text}"}

    def run(self, query):
        return self.invoke(query)["output"]


def create_llm_agent(tools):
    return ParkingLLMAgent(tools)
