import numpy as np


class QLearningModel:
    def __init__(self, zones):
        self.zones = zones
        self.state_count = max(1, len(zones))
        self.q_table = np.zeros((self.state_count, self.state_count))
        self.lr = 0.1
        self.gamma = 0.9

    def get_state(self, observation):
        if not observation:
            return 0
        worst_zone = min(observation, key=lambda item: item["free_slots"])
        return self.zones.index(worst_zone["zone"])

    def choose_action(self, state_index):
        return int(np.argmax(self.q_table[state_index]))

    def update(self, state, action, reward, next_state):
        self.q_table[state][action] += self.lr * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state][action]
        )

    def print_q_table(self):
        print("Q Table")
        print(self.q_table)


QLearningAgent = QLearningModel
