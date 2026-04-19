from llm.fallback import summarize_state

def format_zone_table_lines(state, fields):
    lines = []
    for zone, data in state.items():
        values = [f"{field}: {data.get(field, 0)}" for field in fields]
        lines.append(f"- {zone}: " + ", ".join(values))
    return "\n".join(lines)

def get_local_chat_response(state, query):
    """Provides local heuristics answering chat interfaces automatically."""
    summary = summarize_state(state)
    query_lower = query.lower()

    if any(k in query_lower for k in ["occupied in each", "occupied slots", "all the slots occupied", "zone by zone"]):
        return "Occupied slots by zone:\n" + format_zone_table_lines(state, ["occupied", "free_slots", "total_slots"])

    if "free slots in each" in query_lower or "free slots" in query_lower and "each" in query_lower:
        return "Free slots by zone:\n" + format_zone_table_lines(state, ["free_slots", "occupied", "total_slots"])

    if "entries and exits" in query_lower or "vehicle movement" in query_lower:
        return "Current entries and exits by zone:\n" + format_zone_table_lines(state, ["entry", "exit", "free_slots"])

    if "which zone is full" in query_lower or "fully occupied" in query_lower:
        fullest = max(state, key=lambda zone: state[zone]["occupied"] / max(1, state[zone]["total_slots"]))
        fullness = round((state[fullest]["occupied"] / max(1, state[fullest]["total_slots"])) * 100, 1)
        return f"Based on the latest telemetry, {fullest} is experiencing the most pressure right now at {fullness}% occupancy. There are only {state[fullest]['free_slots']} slots remaining before it reaches full capacity."

    if "current event" in query_lower or "what event" in query_lower:
        return "I can see the event context in the runtime state. If you ask me about the active event alongside the latest allocation, I can give you a combined operational assessment."

    if "slow" in query_lower or "filling" in query_lower:
        slowest = min(state, key=lambda zone: state[zone]["entry"] - state[zone]["exit"])
        return f"Looking at the flow rates, {slowest} seems to be the quietest. It's currently showing relatively low net inflow, with {state[slowest]['free_slots']} free slots still available to soak up demand."

    if "best" in query_lower or "available" in query_lower:
        best = summary["best_zone"]
        return f"For immediate availability, {best} is definitely your best bet. It has a comfortable buffer of {state[best]['free_slots']} free slots right now."

    if "crowd" in query_lower or "full" in query_lower or "congestion" in query_lower:
        crowded = summary["most_crowded"]
        return f"{crowded} is our primary congestion hotspot at the moment, with only {state[crowded]['free_slots']} free slots left. I'd definitely keep an eye on this zone."

    return f"I've scanned the network, and {summary['most_crowded']} is currently the most crowded area. On the flip side, {summary['best_zone']} has the most availability. If congestion persists at {summary['most_crowded']}, I'd recommend preparing a redirect to {summary['best_zone']}."
