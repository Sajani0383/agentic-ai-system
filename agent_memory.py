class AgentMemory:

    def __init__(self):
        self.history = []

    def add(self, state):
        self.history.append(state)

        if len(self.history) > 20:
            self.history.pop(0)

    def get_metrics(self):

        if not self.history:
            return {}

        congestion_count = 0
        total_free = 0
        count = 0

        for state in self.history:
            for zone in state:

                free_slots = state[zone]["free_slots"]

                total_free += free_slots
                count += 1

                if free_slots < 10:
                    congestion_count += 1

        avg_free = total_free / count if count else 0

        return {
            "steps": len(self.history),
            "avg_free_slots": round(avg_free, 2),
            "congestion_events": congestion_count
        }