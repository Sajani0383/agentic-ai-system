class History:

    def __init__(self):
        self.records = []
        self.metrics = []

    def add(self, state):
        self.records.append(state)

        avg_free = sum([d["free_slots"] for d in state.values()]) / len(state)
        congestion = sum([1 for d in state.values() if d["free_slots"] < 10])

        self.metrics.append({
            "avg_free_slots": avg_free,
            "congestion_count": congestion
        })

    def get_trend(self):
        trends = {}
        for state in self.records:
            for zone, data in state.items():
                trends.setdefault(zone, []).append(data["free_slots"])
        return trends

    def get_metrics(self):
        return self.metrics