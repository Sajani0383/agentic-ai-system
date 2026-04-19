import json
import os
import random
from collections import deque
import numpy as np


class QLearningModel:
    PRESSURE_BUCKETS = 4

    def __init__(self, zones, epsilon=0.15, min_epsilon=0.10, epsilon_decay=0.995, seed=7, replay_buffer_size=1000, batch_size=32):
        self.zones = list(zones)
        self.action_space_size = max(1, len(self.zones))
        # Scalable Q-Table using a dictionary mapping a hashed string state -> np.array
        self.q_table = {}
        self.lr = 0.15
        self.gamma = 0.92
        self.epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)
        
        # Experience Replay
        self.memory = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

    def _get_or_create_state(self, state_hash):
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.action_space_size)
        return self.q_table[state_hash]

    def get_state(self, observation):
        """
        Translates a multi-zone observation into a discretized global system state string.
        Enforces normalizations mathematically.
        """
        if not observation or not self.zones:
            return "empty_state"

        zone_states = []
        for item in observation:
            zone_name = item.get("zone")
            if zone_name not in self.zones:
                continue
            
            # Enforce strict Normalization between 0.0 and 1.0
            pressure_norm = max(0.0, min(1.0, float(item.get("pressure_score", 0.0))))
            demand_norm = max(0.0, min(1.0, float(item.get("demand_norm", 0.0))))
            free_slots_norm = max(0.0, min(1.0, float(item.get("free_slots_norm", 1.0))))
            
            p_bucket = min(self.PRESSURE_BUCKETS - 1, int(pressure_norm * self.PRESSURE_BUCKETS))
            d_bucket = min(self.PRESSURE_BUCKETS - 1, int(demand_norm * self.PRESSURE_BUCKETS))
            f_bucket = min(self.PRESSURE_BUCKETS - 1, int(free_slots_norm * self.PRESSURE_BUCKETS))
            
            zone_states.append(f"{zone_name}:{p_bucket}{d_bucket}{f_bucket}")
            
        if not zone_states:
            return "empty_state"
            
        zone_states.sort()
        return "|".join(zone_states)

    def choose_action(self, state_index, explore=True, invalid_actions=None):
        if self.action_space_size <= 1:
            return 0
            
        invalid_actions = invalid_actions or []
        state_q_values = np.copy(self._get_or_create_state(state_index))
        
        # Mask invalid actions
        for idx in invalid_actions:
            if 0 <= idx < self.action_space_size:
                state_q_values[idx] = -np.inf

        valid_actions = [i for i in range(self.action_space_size) if i not in invalid_actions]
        if not valid_actions:
            return 0  # Fallback if everything is invalid

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions)), "EXPLORE"
            
        return int(np.argmax(state_q_values)), "EXPLOIT"

    def action_confidence(self, state_index, temperature=1.0):
        """
        Returns true probabilistic confidence using Softmax on Q-values.
        """
        row = self._get_or_create_state(state_index)
        # Shift values for numerical stability
        shifted = row - np.max(row)
        exp_values = np.exp(shifted / temperature)
        probs = exp_values / np.sum(exp_values)
        return float(round(np.max(probs), 3))

    def update(self, state, action, reward, next_state, done=False):
        """
        Perform 1-step Q-learning update.
        Handles terminal states (done).
        """
        current_q = self._get_or_create_state(state)[action]
        max_next_q = 0.0 if done else np.max(self._get_or_create_state(next_state))
        
        # Internal reward shaping (example): Give tiny bonus for reaching highly valid states
        shaped_reward = reward
        if done and reward > 0:
            shaped_reward += 0.5 

        self.q_table[state][action] = current_q + self.lr * (shaped_reward + self.gamma * max_next_q - current_q)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done):
        """ Add transition to experience replay """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """ Multi-step planning/learning from past experience """
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            # We don't decay epsilon during replay to avoid crushing it too fast
            current_q = self._get_or_create_state(state)[action]
            max_next_q = 0.0 if done else np.max(self._get_or_create_state(next_state))
            self.q_table[state][action] = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

    def print_q_table(self):
        print(f"Q Table (States tracking: {len(self.q_table)})")
        for state, actions in self.q_table.items():
            print(f"{state}: {actions}")
            
    def save(self, filepath):
        """ Serialize Q-table to JSON """
        export_data = {k: v.tolist() for k, v in self.q_table.items()}
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

    def load(self, filepath):
        """ Load Q-table from JSON """
        if not os.path.exists(filepath):
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.q_table = {k: np.array(v) for k, v in data.items()}

QLearningAgent = QLearningModel
