class AgentMemory:

    def __init__(self):
        self.history = []

    def add(self, state):
        self.history.append(state)
        if len(self.history) > 30:
            self.history.pop(0)

    def get_metrics(self):
        metrics = []
        for s in self.history:
            avg = sum(s[z]["free_slots"] for z in s) / len(s)
            congestion = sum(1 for z in s if s[z]["free_slots"] <= 10)

            metrics.append({
                "avg_free_slots": round(avg, 2),
                "congestion_count": congestion
            })
        return metrics