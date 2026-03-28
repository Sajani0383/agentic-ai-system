class BayesianAgent:
    def infer(self, state):
        most_crowded = min(state, key=lambda zone: state[zone]["free_slots"])
        best_zone = max(state, key=lambda zone: state[zone]["free_slots"])
        free_values = [zone["free_slots"] for zone in state.values()]
        spread = max(free_values) - min(free_values)
        confidence = round(min(0.99, 0.55 + spread / 100), 2)

        return {
            "confidence": confidence,
            "most_crowded": most_crowded,
            "best_zone": best_zone,
            "spread": spread,
        }
