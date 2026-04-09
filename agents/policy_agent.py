import numpy as np

from models.q_learning import QLearningAgent


class PolicyAgent:
    def __init__(self, zones=None):
        self.zones = zones or []
        self.q_agent = QLearningAgent(self.zones) if self.zones else None

    def load_q_table(self, q_table):
        if self.q_agent is None or not q_table:
            return
        q_array = np.array(q_table)
        if q_array.shape == self.q_agent.q_table.shape:
            self.q_agent.q_table = q_array

    def export_q_table(self):
        if self.q_agent is None:
            return []
        return self.q_agent.q_table.tolist()

    def decide(self, state, demand, insight):
        from_zone = min(state, key=lambda zone: state[zone]["free_slots"])
        pressure = demand.get(from_zone, 0)
        free_slots = state[from_zone]["free_slots"]

        if free_slots > 12 and pressure < 8:
            return None

        if self.q_agent is not None:
            observation = [
                {"zone": zone, "free_slots": data["free_slots"]}
                for zone, data in state.items()
            ]
            state_index = self.q_agent.get_state(observation)
            destination_index = self.q_agent.choose_action(state_index)
            to_zone = self.zones[destination_index]
        else:
            to_zone = max(state, key=lambda zone: state[zone]["free_slots"])

        if from_zone == to_zone:
            to_zone = max(
                (zone for zone in state if zone != from_zone),
                key=lambda zone: state[zone]["free_slots"],
                default=from_zone,
            )

        if from_zone == to_zone:
            return None

        transferable = max(1, min(10, int((pressure + max(0, 12 - free_slots)) / 2)))

        return {
            "action": "redirect",
            "from": from_zone,
            "to": to_zone,
            "vehicles": transferable,
            "reason": (
                f"RL-informed policy suggests redirecting from {from_zone} to {to_zone} "
                f"because free slots are low and pressure is {pressure}"
            ),
            "confidence": insight.get("confidence", 0.75),
        }

    def update(self, old_state, action, reward, new_state):
        if self.q_agent is None or not action or action.get("action") != "redirect":
            return

        old_observation = [
            {"zone": zone, "free_slots": data["free_slots"]}
            for zone, data in old_state.items()
        ]
        new_observation = [
            {"zone": zone, "free_slots": data["free_slots"]}
            for zone, data in new_state.items()
        ]
        state_index = self.q_agent.get_state(old_observation)
        next_state_index = self.q_agent.get_state(new_observation)
        action_zone = action.get("to")
        if action_zone not in self.zones:
            return
        action_index = self.zones.index(action_zone)
        self.q_agent.update(state_index, action_index, reward, next_state_index)
