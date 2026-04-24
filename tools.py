from langchain.tools import Tool


class NullHistory:
    def __init__(self):
        self.states = []

    def add(self, state):
        self.states.append(state)

    def get_trend(self):
        return self.states

    def get_metrics(self):
        if not self.states:
            return {}
        avg_free = sum(
            zone["free_slots"]
            for state in self.states
            for zone in state.values()
        ) / sum(len(state) for state in self.states)
        return {"steps": len(self.states), "avg_free_slots": round(avg_free, 2)}


def get_tools(environment, history=None, *_unused):
    history = history or NullHistory()

    def get_state(_=None):
        return str(environment.get_state())

    def simulate(_=None):
        state, reward = environment.step()
        history.add(state)
        return f"Simulation updated with reward {reward}:\n{state}"

    def predict(_=None):
        state = environment.get_state()
        result = []
        for zone, data in state.items():
            score = data["entry"] - data["exit"]
            result.append(f"{zone}: demand pressure = {score}")
        return "\n".join(result)

    def decision(_=None):
        state = environment.get_state()
        best_zone = max(state, key=lambda zone: state[zone]["free_slots"])
        crowded_zone = min(state, key=lambda zone: state[zone]["free_slots"])
        return (
            f"Best zone: {best_zone}\n"
            f"Most crowded zone: {crowded_zone}\n"
            f"Reason: choose zones with more free slots and lower congestion."
        )

    def trend(_=None):
        return str(history.get_trend())

    def metrics(_=None):
        return str(history.get_metrics())

    return [
        Tool(name="Get State", func=get_state, description="Current parking state"),
        Tool(name="Simulate", func=simulate, description="Run simulation"),
        Tool(name="Predict Demand", func=predict, description="Demand analysis"),
        Tool(name="Decision", func=decision, description="Best action"),
        Tool(name="Trend", func=trend, description="Trend analysis"),
        Tool(name="Metrics", func=metrics, description="Performance metrics"),
    ]


def build_runtime_tools(environment, memory):
    def get_state_snapshot():
        return environment.get_state()

    def get_goal_status():
        return memory.get_active_goal()

    def get_recent_cycles():
        return memory.get_recent_cycles(limit=5)

    def get_memory_metrics():
        return memory.get_metrics()

    def get_learning_profile(scenario_mode=None, from_zone=None, to_zone=None):
        return memory.get_learning_profile(
            scenario_mode=scenario_mode,
            from_zone=from_zone,
            to_zone=to_zone,
        )

    def get_event_context():
        return environment.get_event_context()

    def get_operational_signals():
        return environment.get_operational_signals()

    def get_scenario_mode():
        return environment.get_scenario_mode()

    def estimate_transfer_capacity(from_zone, to_zone, requested):
        state = environment.get_state()
        if from_zone not in state or to_zone not in state:
            return 0
        available = max(
            state[from_zone]["entry"],
            max(0, 12 - state[from_zone]["free_slots"]),
        )
        free_capacity = state[to_zone]["free_slots"]
        return max(0, min(requested, available, free_capacity))

    def build_zone_pressure_report(state, demand):
        return {
            zone: {
                "free_slots": state[zone]["free_slots"],
                "occupied": state[zone]["occupied"],
                "demand_pressure": demand.get(zone, 0),
            }
            for zone in state
        }

    def suggest_best_zone(state):
        return max(
            state,
            key=lambda zone: (state[zone]["free_slots"], -state[zone]["occupied"]),
        )

    def get_zone_status(zone_name):
        return environment.get_state().get(zone_name, {})

    def build_belief_state(state, demand, insight=None):
        insight = insight or {}
        learning = memory.get_learning_profile()
        blocked_routes = set(learning.get("blocked_routes", []))
        zones = {}
        for zone, data in state.items():
            free_slots = int(data.get("free_slots", 0) or 0)
            demand_pressure = float(demand.get(zone, 0) or 0)
            posterior = insight.get("posteriors", {}).get(zone)
            if posterior is None:
                posterior = insight.get("normalized_posteriors", {}).get(zone, 0.0)
            zones[zone] = {
                "risk": round(max(0.0, min(1.0, (12 - free_slots) / 12 + demand_pressure / 120 + float(posterior or 0.0) * 0.25)), 3),
                "free_slots": free_slots,
                "demand_pressure": demand_pressure,
                "trusted_destination": free_slots >= 4,
            }
        return {
            "zones": zones,
            "blocked_routes": sorted(blocked_routes),
            "reward_trend": learning.get("recent_reward_avg", 0.0),
            "none_block_active": learning.get("none_block_active", False),
        }

    def route_risk_check(from_zone, to_zone):
        learning = memory.get_learning_profile(from_zone=from_zone, to_zone=to_zone)
        route_key = f"{from_zone}->{to_zone}"
        return {
            "route": route_key,
            "blocked": route_key in learning.get("blocked_routes", []),
            "consecutive_failures": learning.get("route_consecutive_failures", {}).get(route_key, 0),
            "route_profile": learning.get("route_profile", {}),
        }

    def reward_trend_analysis():
        learning = memory.get_learning_profile()
        avg_reward = float(learning.get("recent_reward_avg", 0.0) or 0.0)
        if avg_reward < -0.2:
            direction = "negative"
        elif avg_reward > 0.15:
            direction = "positive"
        else:
            direction = "stable"
        return {
            "direction": direction,
            "recent_reward_avg": avg_reward,
            "last_reward": learning.get("last_reward", 0.0),
            "recommended_transfer_bias": learning.get("global_transfer_bias", 1.0),
        }

    return {
        "get_state_snapshot": get_state_snapshot,
        "get_goal_status": get_goal_status,
        "get_recent_cycles": get_recent_cycles,
        "get_memory_metrics": get_memory_metrics,
        "get_learning_profile": get_learning_profile,
        "get_event_context": get_event_context,
        "get_operational_signals": get_operational_signals,
        "get_scenario_mode": get_scenario_mode,
        "estimate_transfer_capacity": estimate_transfer_capacity,
        "build_zone_pressure_report": build_zone_pressure_report,
        "suggest_best_zone": suggest_best_zone,
        "get_zone_status": get_zone_status,
        "build_belief_state": build_belief_state,
        "route_risk_check": route_risk_check,
        "reward_trend_analysis": reward_trend_analysis,
    }
