class PolicyAgent:

    def decide(self, state, demand, insight):

        worst = min(state, key=lambda z: state[z]["free_slots"])

        if state[worst]["free_slots"] <= 10:
            return {
                "type": "redirect",
                "from": worst,
                "amount": int(demand * 0.2)
            }

        return None