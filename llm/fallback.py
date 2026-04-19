def summarize_state(state):
    """Produces unified crowding heuristics."""
    crowded = min(state, key=lambda zone: state[zone]["free_slots"])
    best = max(state, key=lambda zone: state[zone]["free_slots"])
    return {
        "most_crowded": crowded,
        "best_zone": best,
        "crowded_free_slots": state[crowded]["free_slots"],
        "best_free_slots": state[best]["free_slots"],
    }

def build_advanced_fallback(state, demand=None, insight=None):
    """
    Intelligent deterministic logic utilized if LLM hangs or crashes.
    Utilizes demand overlays to aggressively route.
    """
    summary = summarize_state(state)
    crowded = summary["most_crowded"]
    best = summary["best_zone"]
    
    # Defaults
    demand = demand or {}
    velocities = insight.get("flow_velocities", {}) if insight else {}

    # Aggressive redirection if demand is incoming strongly
    # DemandAgent produces flat {zone: int}; guard against nested-dict shape too.
    raw_demand = demand.get(crowded, 0)
    incoming_pressure = raw_demand.get("inflow", 0) if isinstance(raw_demand, dict) else int(raw_demand or 0)
    current_free = state[crowded]["free_slots"]
    
    if current_free <= 10 and crowded != best:
        # Scale vehicles to shift based on inflow momentum
        vehicles = max(1, min(8, (12 - current_free)))
        if incoming_pressure > 5:
            vehicles += 2 # Aggressively bounce demand
            
        return {
            "action": "redirect",
            "from": crowded,
            "to": best,
            "vehicles": min(vehicles, 10),
            "reason": f"{crowded} is nearing critical congestion limits ({current_free} slots free) with elevated incoming pressure ({incoming_pressure} vehicles). {best} has substantial reserve capacity. Redirecting vehicles structurally reduces immediate bottleneck risk.",
            "confidence": 0.92,
            "rationale": f"Calculated high risk at {crowded}. LLM fallback simulated a strategic redirect to {best} to preserve network flow."
        }

    return {
        "action": "none",
        "from": crowded,
        "to": best,
        "vehicles": 0,
        "reason": f"Network pressure is currently stable. {crowded} is our most utilized zone but maintains a safe buffer of {current_free} slots. Holding baseline configuration to avoid unnecessary routing friction.",
        "confidence": 0.88,
        "rationale": "Calculated low systemic risk. LLM fallback suggests maintaining current routing policy."
    }
