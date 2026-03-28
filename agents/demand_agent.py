class DemandAgent:
    def predict(self, state):
        demand = {}

        for zone, data in state.items():
            pressure = max(0, data["entry"] - data["exit"])
            occupancy_ratio = (
                data["occupied"] / data["total_slots"] if data["total_slots"] else 0
            )
            scarcity = max(0, int((0.85 - (1 - occupancy_ratio)) * 40))
            demand[zone] = min(100, pressure + scarcity)

        return demand
