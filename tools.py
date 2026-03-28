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
