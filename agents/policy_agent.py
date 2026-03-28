class PolicyAgent:
    def __init__(self, zones=None):
        self.zones = zones or []

    def decide(self, state, demand, insight):
        from_zone = min(state, key=lambda zone: state[zone]["free_slots"])
        to_zone = max(state, key=lambda zone: state[zone]["free_slots"])

        if from_zone == to_zone:
            return None

        pressure = demand.get(from_zone, 0)
        free_slots = state[from_zone]["free_slots"]

        if free_slots > 12 and pressure < 8:
            return None

        transferable = max(1, min(10, int((pressure + max(0, 12 - free_slots)) / 2)))

        return {
            "action": "redirect",
            "from": from_zone,
            "to": to_zone,
            "vehicles": transferable,
            "reason": (
                f"Redirect from {from_zone} to {to_zone} "
                f"because free slots are low and pressure is {pressure}"
            ),
            "confidence": insight.get("confidence", 0.75),
        }
