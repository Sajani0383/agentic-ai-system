import numpy as np


class QLearningModel:
    PRESSURE_BUCKETS = 4

    def __init__(self, zones, epsilon=0.12, min_epsilon=0.02, epsilon_decay=0.995, seed=7):
        self.zones = list(zones)
        self.state_count = max(1, len(self.zones) * self.PRESSURE_BUCKETS)
        self.q_table = np.zeros((self.state_count, max(1, len(self.zones))))
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)

    def get_state(self, observation):
        if not observation or not self.zones:
            return 0

        filtered = [item for item in observation if item.get("zone") in self.zones]
        if not filtered:
            return 0

        worst_zone = max(
            filtered,
            key=lambda item: (
                item.get("pressure_score", 0.0),
                item.get("demand_norm", 0.0),
                -item.get("free_slots_norm", 1.0),
            ),
        )
        zone_index = self.zones.index(worst_zone["zone"])
        pressure_bucket = min(
            self.PRESSURE_BUCKETS - 1,
            max(0, int(worst_zone.get("pressure_score", 0.0) * self.PRESSURE_BUCKETS)),
        )
        return zone_index * self.PRESSURE_BUCKETS + pressure_bucket

    def choose_action(self, state_index, explore=True):
        if self.q_table.shape[1] <= 1:
            return 0
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state_index]))

    def action_confidence(self, state_index):
        row = self.q_table[state_index]
        if row.size == 0:
            return 0.0
        max_q = float(np.max(row))
        min_q = float(np.min(row))
        spread = max_q - min_q
        return round(max(0.05, min(0.99, 0.5 + spread / max(1.0, abs(max_q) + 1.0))), 3)

    def update(self, state, action, reward, next_state):
        self.q_table[state][action] += self.lr * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state][action]
        )
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def print_q_table(self):
        print("Q Table")
        print(self.q_table)


QLearningAgent = QLearningModel
