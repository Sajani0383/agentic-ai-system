import random
from datetime import datetime
from ml.predict import predict_demand

class ParkingEnvironment:

    def __init__(self):
        self.zones = ["Mall", "Hospital", "Office", "Residential", "Commercial"]
        self.zone_map = {z: i for i, z in enumerate(self.zones)}

        self.state = {
            z: {
                "total_slots": random.randint(80, 120),
                "occupied": random.randint(20, 50),
                "entry": 0,
                "exit": 0
            }
            for z in self.zones
        }

        self.history = []

    def step(self):
        now = datetime.now()
        hour = now.hour
        day = now.day

        snapshot = {}

        for z in self.zones:
            zone_id = self.zone_map[z]

            demand = max(20, predict_demand(hour, day, zone_id, 0))

            entry = int(demand * random.uniform(0.2, 0.4))
            exit = int(entry * random.uniform(0.3, 0.6))

            entry += random.randint(-5, 5)
            exit += random.randint(-3, 3)

            entry = max(0, entry)
            exit = max(0, exit)

            self.state[z]["entry"] = entry
            self.state[z]["exit"] = exit

            self.state[z]["occupied"] += entry - exit

            self.state[z]["occupied"] = max(0, self.state[z]["occupied"])
            self.state[z]["occupied"] = min(
                self.state[z]["occupied"],
                self.state[z]["total_slots"]
            )

            snapshot[z] = self.state[z]["total_slots"] - self.state[z]["occupied"]

        self.history.append(snapshot)

        if len(self.history) > 30:
            self.history.pop(0)

        return self.get_state()

    def apply_action(self, action):
        if action and action["type"] == "redirect":
            z = action["from"]
            self.state[z]["occupied"] -= action["amount"]
            self.state[z]["occupied"] = max(0, self.state[z]["occupied"])

    def get_state(self):
        return {
            z: {
                "free_slots": self.state[z]["total_slots"] - self.state[z]["occupied"],
                "entry": self.state[z]["entry"],
                "exit": self.state[z]["exit"]
            }
            for z in self.zones
        }

    def get_trend(self):
        return self.history