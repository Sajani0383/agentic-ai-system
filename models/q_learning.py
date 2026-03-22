import numpy as np

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((5, 5))
        self.lr = 0.1
        self.gamma = 0.9

    def choose_action(self, state_index):
        return np.argmax(self.q_table[state_index])

    def update(self, state, action, reward, next_state):
        self.q_table[state][action] += self.lr * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )