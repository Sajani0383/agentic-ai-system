import json
import os

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


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


def get_llm():
    if os.getenv("ENABLE_LLM", "false").lower() not in {"1", "true", "yes"}:
        return None

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or ChatGoogleGenerativeAI is None:
        return None

    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=api_key,
        )
    except Exception:
        return None


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
        if "```" in response:
            response = response.split("```")[1]
        response = response.replace("json", "").strip()
        start = response.find("{")
        end = response.rfind("}") + 1
        decision = json.loads(response[start:end])
        if "action" not in decision:
            return fallback
        return decision
    except Exception:
        return fallback


def get_local_chat_response(state, query):
    summary = summarize_state(state)
    query_lower = query.lower()

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
