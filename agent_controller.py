class AgentMemory:

    def __init__(self):
        self.history = []

    def add(self, state):
        self.history.append(state)

        if len(self.history) > 20:
            self.history.pop(0)

    def get_metrics(self):
        metrics = []

        for snapshot in self.history:
            total_free = sum(snapshot[z]["free_slots"] for z in snapshot)
            avg_free = total_free / len(snapshot)

            # Better congestion detection
            congestion = sum(
                1 for z in snapshot if snapshot[z]["free_slots"] <= 10
            )

            metrics.append({
                "avg_free_slots": round(avg_free, 2),
                "congestion_count": congestion
            })

        return metrics