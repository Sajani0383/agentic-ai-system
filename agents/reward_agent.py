class RewardAgent:
    def evaluate(self, old_state, new_state):
        old_total = sum(old_state[z]["free_slots"] for z in old_state)
        new_total = sum(new_state[z]["free_slots"] for z in new_state)
        return new_total - old_total