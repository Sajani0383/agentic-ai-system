class RewardAgent:
    DEFAULTS = {
        "base_congestion_ratio": 0.12,
        "movement_cost_per_vehicle": 0.06,
        "unnecessary_action_penalty": 0.45,
        "event_focus_bonus": 0.35,
        "recommended_zone_bonus": 0.18,
        "queue_penalty_weight": 0.08,
        "search_time_penalty_weight": 0.12,
        "delayed_reward_weight": 0.18,
        "stagnation_penalty": 0.08,
    }

    def evaluate(
        self,
        old_state,
        new_state,
        action=None,
        demand=None,
        event_context=None,
        kpis=None,
        transition=None,
    ):
        demand = demand or {}
        event_context = event_context or {}
        kpis = kpis or {}
        transition = transition or {}

        old_metrics = self._state_metrics(old_state, demand, event_context)
        new_metrics = self._state_metrics(new_state, demand, event_context)
        zone_count = max(1, len(new_state or {}))

        congestion_gain = (old_metrics["normalized_pressure"] - new_metrics["normalized_pressure"]) * 2.2
        demand_relief = old_metrics["demand_weighted_pressure"] - new_metrics["demand_weighted_pressure"]
        balance_gain = old_metrics["imbalance"] - new_metrics["imbalance"]
        priority_gain = new_metrics["priority_score"] - old_metrics["priority_score"]

        action_type = (action or {}).get("action", "none")
        moved = transition.get("transfer_detail", {}).get("moved", 0) if isinstance(transition, dict) else 0
        requested = transition.get("transfer_detail", {}).get("requested", 0) if isinstance(transition, dict) else 0
        movement_cost = moved * self.DEFAULTS["movement_cost_per_vehicle"]
        unnecessary_penalty = 0.0
        if action_type == "redirect" and moved == 0:
            unnecessary_penalty += self.DEFAULTS["unnecessary_action_penalty"]
        elif action_type == "redirect" and congestion_gain <= 0 and demand_relief <= 0:
            unnecessary_penalty += self.DEFAULTS["unnecessary_action_penalty"] * 0.7

        queue_penalty = kpis.get("queue_length", 0) * self.DEFAULTS["queue_penalty_weight"]
        search_time_penalty = max(0.0, kpis.get("estimated_search_time_min", 0.0) - 2.0) * self.DEFAULTS["search_time_penalty_weight"]
        delayed_impact = (
            kpis.get("allocation_success_pct", 100.0) / 100.0
            - min(1.0, kpis.get("congestion_hotspots", 0) / max(1, zone_count))
        ) * self.DEFAULTS["delayed_reward_weight"]

        base_reward = congestion_gain + demand_relief + balance_gain * 0.8 + priority_gain + delayed_impact
        reward = base_reward - movement_cost - unnecessary_penalty - queue_penalty - search_time_penalty

        if abs(reward) < 0.05:
            reward -= self.DEFAULTS["stagnation_penalty"]

        if requested > moved and action_type == "redirect":
            reward -= min(0.4, (requested - moved) / max(1, requested))

        return round(reward, 3)

    def _state_metrics(self, state, demand, event_context):
        if not state:
            return {
                "normalized_pressure": 0.0,
                "demand_weighted_pressure": 0.0,
                "imbalance": 0.0,
                "priority_score": 0.0,
            }

        zone_pressures = []
        demand_pressures = []
        free_slots = []
        priority_score = 0.0
        max_demand = max([demand.get(zone, 0) for zone in state] or [1])
        focus_zone = event_context.get("focus_zone")
        recommended_zone = event_context.get("recommended_zone")

        for zone, data in state.items():
            total_slots = max(1, data.get("total_slots", 1))
            free_slots_count = max(0, min(total_slots, data.get("free_slots", total_slots - data.get("occupied", 0))))
            adaptive_threshold = max(6, int(round(total_slots * self.DEFAULTS["base_congestion_ratio"])))
            free_ratio = free_slots_count / total_slots
            pressure = max(0.0, (adaptive_threshold - free_slots_count) / max(1, adaptive_threshold))
            demand_weight = demand.get(zone, 0) / max(1, max_demand)
            zone_pressures.append(pressure)
            demand_pressures.append(pressure * (0.6 + demand_weight))
            free_slots.append(free_ratio)

            if zone == focus_zone:
                priority_score += free_ratio * self.DEFAULTS["event_focus_bonus"]
            if zone == recommended_zone:
                priority_score += free_ratio * self.DEFAULTS["recommended_zone_bonus"]

        imbalance = (max(free_slots) - min(free_slots)) if free_slots else 0.0
        return {
            "normalized_pressure": sum(zone_pressures) / len(zone_pressures),
            "demand_weighted_pressure": sum(demand_pressures) / len(demand_pressures),
            "imbalance": imbalance,
            "priority_score": priority_score,
        }
