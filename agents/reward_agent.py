class RewardAgent:
    def evaluate(self, old_state, new_state):
        old_pressure = sum(max(0, 10 - old_state[zone]["free_slots"]) for zone in old_state)
        new_pressure = sum(max(0, 10 - new_state[zone]["free_slots"]) for zone in new_state)
        return round(old_pressure - new_pressure, 2)
