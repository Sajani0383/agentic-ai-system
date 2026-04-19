from copy import deepcopy

import numpy as np

from adk.trace_logger import trace_logger
from models.q_learning import QLearningAgent


class PolicyAgent:
    def __init__(self, zones=None, logger=None):
        self.zones = list(zones or [])
        self.logger = logger or trace_logger
        self.q_agent = QLearningAgent(self.zones) if self.zones else None
        self.last_decision = {}

    def load_q_table(self, q_table):
        if self.q_agent is None or not q_table:
            return
        if isinstance(q_table, dict):
            self.q_agent.q_table = {k: np.array(v) for k, v in q_table.items()}

    def export_q_table(self):
        if self.q_agent is None:
            return {}
        return {k: v.tolist() for k, v in self.q_agent.q_table.items()}

    def decide(self, state, demand, insight, event_context=None, learning_profile=None):
        step = "-"
        if not isinstance(state, dict) or not state:
            decision = self._noop_decision("PolicyAgent received an empty or invalid state.")
            self._record_decision(step, decision)
            return decision

        active_zones = [zone for zone in self.zones if zone in state] or list(state.keys())
        if len(active_zones) < 2:
            decision = self._noop_decision("At least two valid zones are required for policy routing.")
            self._record_decision(step, decision)
            return decision

        state_view = {zone: state[zone] for zone in active_zones}
        event_context = event_context or {}
        source_zone = min(
            state_view,
            key=lambda zone: (
                state_view[zone]["free_slots"],
                -demand.get(zone, 0),
            ),
        )
        observation = self._build_observation(state_view, demand, insight, event_context)
        rl_state_index = self.q_agent.get_state(observation) if self.q_agent is not None else 0
        source_index = self.zones.index(source_zone) if source_zone in self.zones else None
        invalid_actions = [source_index] if source_index is not None else []
        
        rl_action_index = None
        decision_mode = "AUTONOMOUS_POLICY"
        
        # ── Step A: Recovery Loop Logic ──
        # If reward is crashing, shift behavior to safety/stabilization
        learning_applied = False
        if learning_profile:
            recent_reward = learning_profile.get("recent_reward_avg", 0.0)
            if recent_reward < -0.4:
                decision_mode = "ADAPTIVE_STABILIZATION"
                # Force a reduction in exploration to focus on known safer patterns
                if self.q_agent:
                    self.q_agent.epsilon = max(0.05, self.q_agent.epsilon * 0.5)
                learning_applied = True

        if self.q_agent is not None:
            rl_action_index, choose_mode = self.q_agent.choose_action(rl_state_index, explore=True, invalid_actions=invalid_actions)
            if decision_mode != "ADAPTIVE_STABILIZATION":
                decision_mode = choose_mode
            
        rl_zone = self._zone_for_index(rl_action_index, active_zones)
        fallback_zone = max(
            (zone for zone in state_view if zone != source_zone),
            key=lambda zone: (
                state_view[zone]["free_slots"],
                -demand.get(zone, 0),
            ),
            default=source_zone,
        )
        destination_zone = rl_zone if rl_zone and rl_zone != source_zone else fallback_zone
        if destination_zone == source_zone:
            decision = self._noop_decision("Policy baseline found no safe alternate destination.")
            self._record_decision(step, decision)
            return decision

        source_data = state_view[source_zone]
        pressure = demand.get(source_zone, 0)
        congestion_gap = max(0, 12 - source_data["free_slots"])
        transferable = max(1, min(12, int(round((pressure / 10) + congestion_gap / 2 + source_data["entry"] / 2))))
        route_certainty = self.q_agent.action_confidence(rl_state_index) if self.q_agent is not None else 0.6
        uncertainty_penalty = min(0.18, float(insight.get("uncertainty", {}).get("entropy", 0.0)) / 10)
        confidence = round(max(0.3, min(0.95, route_certainty - uncertainty_penalty)), 3)
        
        # ── Step B: Safety Overrides ──
        # If in recovery mode, higher hold probability
        hold_bonus = 3 if decision_mode == "ADAPTIVE_STABILIZATION" else 0
        hold_threshold = source_data["free_slots"] >= (14 - hold_bonus) and pressure < (8 - hold_bonus) and source_data["entry"] <= source_data["exit"] + 1

        if hold_threshold:
            decision = {
                "action": "none",
                "from": source_zone,
                "to": destination_zone,
                "vehicles": 0,
                "reason": (
                    f"RL baseline evaluated route {source_zone} -> {destination_zone}, but current pressure is low enough to hold."
                ),
                "confidence": confidence,
                "policy_source": "hybrid_rl_rule_policy",
                "rl_state_index": rl_state_index,
                "rl_destination": destination_zone,
                "exploration_mode": decision_mode,
                "exploration_rate": round(self.q_agent.epsilon, 3) if self.q_agent is not None else 0.0,
            }
            self._record_decision(step, decision)
            return decision

        decision = {
            "action": "redirect",
            "from": source_zone,
            "to": destination_zone,
            "vehicles": transferable,
            "reason": (
                f"RL-informed baseline selected {destination_zone} for {source_zone} using normalized congestion, demand, "
                f"and event-aware pressure features."
            ),
            "confidence": confidence,
            "policy_source": "hybrid_rl_rule_policy",
            "rl_state_index": rl_state_index,
            "rl_destination": destination_zone,
            "exploration_mode": decision_mode,
            "exploration_rate": round(self.q_agent.epsilon, 3) if self.q_agent is not None else 0.0,
            "learning_applied": learning_applied,
        }
        self._record_decision(step, decision)
        return decision

    def update(self, old_state, action, reward, new_state, demand=None, insight=None, execution_feedback=None, agent_memory=None):
        if self.q_agent is None or not isinstance(old_state, dict) or not old_state or not isinstance(new_state, dict) or not new_state:
            return

        demand = demand or {}
        insight = insight or {}
        execution_feedback = execution_feedback or {}
        old_active = {zone: old_state[zone] for zone in self.zones if zone in old_state} or old_state
        new_active = {zone: new_state[zone] for zone in self.zones if zone in new_state} or new_state
        old_observation = self._build_observation(old_active, demand, insight, {})
        new_observation = self._build_observation(new_active, demand, insight, {})
        state_index = self.q_agent.get_state(old_observation)
        next_state_index = self.q_agent.get_state(new_observation)

        action_zone = None
        if action and action.get("action") == "redirect":
            action_zone = action.get("to")
        elif execution_feedback.get("blocked_action", {}).get("action") == "redirect":
            action_zone = execution_feedback["blocked_action"].get("to")
            reward -= 1.5

        if action_zone not in self.zones:
            return

        action_index = self.zones.index(action_zone)
        if execution_feedback and not execution_feedback.get("success", True):
            reward = min(reward - 1.0, -1.0)
            
        done = False # A true terminal check can be added here
        self.q_agent.update(state_index, action_index, reward, next_state_index, done=done)
        self.q_agent.remember(state_index, action_index, reward, next_state_index, done)
        self.q_agent.replay()
        self.logger.log(
            "-",
            "policy_learning_update",
            {
                "learning_profile": agent_memory.get_learning_profile() if agent_memory else {},
                "next_state_index": next_state_index,
                "action_zone": action_zone,
                "reward": round(reward, 3),
                "epsilon": round(self.q_agent.epsilon, 3),
            },
            level="INFO",
        )

    def get_last_decision(self):
        return deepcopy(self.last_decision)

    def _build_observation(self, state, demand, insight, event_context):
        max_demand = max([demand.get(zone, 0) for zone in state] or [1])
        event_focus = event_context.get("focus_zone")
        recommended_zone = event_context.get("recommended_zone")
        normalized = []
        for zone, data in state.items():
            total_slots = max(1, data.get("total_slots", 1))
            free_norm = round(data.get("free_slots", 0) / total_slots, 4)
            demand_norm = round(demand.get(zone, 0) / max(1, max_demand), 4)
            pressure_score = round(
                min(
                    1.0,
                    (1 - free_norm) * 0.45
                    + demand_norm * 0.30
                    + min(1.0, max(0, data.get("entry", 0) - data.get("exit", 0)) / total_slots * 6) * 0.15
                    + (0.06 if zone == event_focus else 0.0)
                    + (0.04 if zone == recommended_zone else 0.0),
                ),
                4,
            )
            normalized.append(
                {
                    "zone": zone,
                    "free_slots": data.get("free_slots", 0),
                    "free_slots_norm": free_norm,
                    "demand_norm": demand_norm,
                    "entry_norm": round(data.get("entry", 0) / total_slots, 4),
                    "exit_norm": round(data.get("exit", 0) / total_slots, 4),
                    "uncertainty": round(float(insight.get("uncertainty", {}).get("entropy", 0.0)), 4),
                    "pressure_score": pressure_score,
                }
            )
        return normalized

    def _zone_for_index(self, action_index, active_zones):
        if action_index is None or self.q_agent is None:
            return None
        if 0 <= action_index < len(self.zones):
            candidate = self.zones[action_index]
            if candidate in active_zones:
                return candidate
        return None

    def _noop_decision(self, reason):
        return {
            "action": "none",
            "from": None,
            "to": None,
            "vehicles": 0,
            "reason": reason,
            "confidence": 0.2,
            "policy_source": "hybrid_rl_rule_policy",
        }

    def _record_decision(self, step, decision):
        self.last_decision = deepcopy(decision)
        self.logger.log(step, "policy_decision", self.last_decision, level="INFO")
