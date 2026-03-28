import random
from datetime import datetime

from ml.predict import predict_demand


class ParkingEnvironment:
    def __init__(self, zones=None):
        self.zones = zones or [
            "Mall",
            "Hospital",
            "Office",
            "Residential",
            "Commercial",
        ]
        self.zone_map = {zone: index for index, zone in enumerate(self.zones)}
        self.history = []
        self.reset()

    def reset(self):
        self.state = {
            zone: {
                "total_slots": random.randint(80, 120),
                "occupied": random.randint(20, 50),
                "entry": 0,
                "exit": 0,
            }
            for zone in self.zones
        }
        self.history = [self.get_state()]
        return self.get_state()

    def step(self, action=None):
        previous_state = self.get_state()

        if action:
            self.apply_action(action)

        now = datetime.now()
        hour = now.hour
        day = now.day

        for zone in self.zones:
            zone_id = self.zone_map[zone]
            demand = max(0, predict_demand(hour, day, zone_id, 0))

            total = self.state[zone]["total_slots"]
            occupied = self.state[zone]["occupied"]
            occupancy_ratio = occupied / total if total else 0
            free_capacity = max(0, total - occupied)

            if occupancy_ratio > 0.85:
                entry = int(demand * random.uniform(0.05, 0.15))
            elif occupancy_ratio > 0.65:
                entry = int(demand * random.uniform(0.10, 0.25))
            else:
                entry = int(demand * random.uniform(0.15, 0.35))

            if occupancy_ratio > 0.75:
                exit_count = int(occupied * random.uniform(0.20, 0.35))
            else:
                exit_count = int(occupied * random.uniform(0.08, 0.20))

            entry = min(max(0, entry), max(0, free_capacity + exit_count))
            exit_count = min(max(0, exit_count), occupied)

            self.state[zone]["entry"] = max(0, entry)
            self.state[zone]["exit"] = max(0, exit_count)
            self.state[zone]["occupied"] += self.state[zone]["entry"] - self.state[zone]["exit"]
            self.state[zone]["occupied"] = max(0, min(self.state[zone]["occupied"], total))

        new_state = self.get_state()
        reward = self._calculate_reward(previous_state, new_state, action)

        self.history.append(new_state)
        if len(self.history) > 50:
            self.history.pop(0)

        return new_state, reward

    def apply_action(self, action):
        if action.get("action") != "redirect":
            return

        from_zone = action.get("from")
        to_zone = action.get("to")
        vehicles = int(action.get("vehicles", 0))

        if from_zone not in self.state or to_zone not in self.state or from_zone == to_zone:
            return

        available_to_move = min(vehicles, self.state[from_zone]["occupied"])
        free_capacity = self.state[to_zone]["total_slots"] - self.state[to_zone]["occupied"]
        transfer = max(0, min(available_to_move, free_capacity))

        if transfer == 0:
            return

        self.state[from_zone]["occupied"] -= transfer
        self.state[to_zone]["occupied"] += transfer

    def get_state(self):
        return {
            zone: {
                "total_slots": self.state[zone]["total_slots"],
                "occupied": self.state[zone]["occupied"],
                "free_slots": self.state[zone]["total_slots"] - self.state[zone]["occupied"],
                "entry": self.state[zone]["entry"],
                "exit": self.state[zone]["exit"],
            }
            for zone in self.zones
        }

    def get_trend(self):
        return self.history

    def _calculate_reward(self, previous_state, new_state, action):
        previous_pressure = sum(max(0, 10 - zone["free_slots"]) for zone in previous_state.values())
        new_pressure = sum(max(0, 10 - zone["free_slots"]) for zone in new_state.values())
        balance_bonus = -(
            max(zone["occupied"] for zone in new_state.values())
            - min(zone["occupied"] for zone in new_state.values())
        ) / 20
        action_bonus = 1 if action and action.get("action") == "redirect" else 0
        return round((previous_pressure - new_pressure) + balance_bonus + action_bonus, 2)
