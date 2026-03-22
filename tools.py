from langchain.tools import Tool

def get_tools(environment, history):

    def get_state(_=None):
        return str(environment.get_state())

    def simulate(_=None):
        state = environment.step()
        history.add(state)
        return f"Simulation updated:\n{state}"

    def predict(_=None):
        state = environment.get_state()
        result = []
        for zone, data in state.items():
            score = data["entry"] - data["exit"]
            result.append(f"{zone}: demand pressure = {score}")
        return "\n".join(result)

    def decision(_=None):
        state = environment.get_state()

        best_zone = None
        best_score = float("inf")
        explanation = []

        for zone, data in state.items():
            pressure = data["entry"] - data["exit"]

            explanation.append(
                f"{zone}: free={data['free_slots']} pressure={pressure}"
            )

            if pressure < best_score:
                best_score = pressure
                best_zone = zone

        confidence = round(1 / (1 + best_score), 2)

        return (
            f"Best zone: {best_zone}\n"
            f"Confidence: {confidence}\n"
            f"Reason:\n" + "\n".join(explanation)
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