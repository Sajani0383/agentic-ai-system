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
    }
