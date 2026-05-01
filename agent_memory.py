import json
import logging
import os
import threading
from copy import deepcopy
from datetime import datetime


class MemoryMetricsEngine:
    """Tracks O(1) incremental statistics for extreme performance bounds."""
    def __init__(self):
        self.steps = 0
        self.total_free_slots_snapshot = 0.0
        self.congestion_events = 0
        self.utilisation_sum = 0.0
        self.search_time_sum = 0.0
        self.allocation_success_sum = 0.0
        self.kpi_count = 0
        
        self.reward_sum = 0.0
        self.reward_count = 0
        
        self.goal_updates = 0

    def add_step(self, free_slots, is_congested):
        self.steps += 1
        self.total_free_slots_snapshot = free_slots 
        if is_congested:
            self.congestion_events += 1

    def add_kpis(self, kpis):
        if kpis:
            self.utilisation_sum += kpis.get("space_utilisation_pct", 0.0)
            self.search_time_sum += kpis.get("estimated_search_time_min", 0.0)
            self.allocation_success_sum += kpis.get("allocation_success_pct", 0.0)
            self.kpi_count += 1

    def add_reward(self, reward_score):
        self.reward_sum += reward_score
        self.reward_count += 1

    def get_aggregated_metrics(self):
        return {
            "steps": self.steps,
            "avg_free_slots": round(self.total_free_slots_snapshot, 2),
            "congestion_events": self.congestion_events,
            "avg_space_utilisation_pct": round(self.utilisation_sum / self.kpi_count, 2) if self.kpi_count else 0.0,
            "avg_search_time_min": round(self.search_time_sum / self.kpi_count, 2) if self.kpi_count else 0.0,
            "allocation_success_pct": round(self.allocation_success_sum / self.kpi_count, 2) if self.kpi_count else 0.0,
            "avg_reward_score": round(self.reward_sum / self.reward_count, 2) if self.reward_count else 0.0,
            "goal_updates": self.goal_updates,
        }

    def load_from_payload(self, state):
        self.steps = state.get("steps", 0)
        self.total_free_slots_snapshot = state.get("total_free_slots_snapshot", 0.0)
        self.congestion_events = state.get("congestion_events", 0)
        self.utilisation_sum = state.get("utilisation_sum", 0.0)
        self.search_time_sum = state.get("search_time_sum", 0.0)
        self.allocation_success_sum = state.get("allocation_success_sum", 0.0)
        self.kpi_count = state.get("kpi_count", 0)
        self.reward_sum = state.get("reward_sum", 0.0)
        self.reward_count = state.get("reward_count", 0)
        self.goal_updates = state.get("goal_updates", 0)

    def export(self):
        return {
            "steps": self.steps,
            "total_free_slots_snapshot": self.total_free_slots_snapshot,
            "congestion_events": self.congestion_events,
            "utilisation_sum": self.utilisation_sum,
            "search_time_sum": self.search_time_sum,
            "allocation_success_sum": self.allocation_success_sum,
            "kpi_count": self.kpi_count,
            "reward_sum": self.reward_sum,
            "reward_count": self.reward_count,
            "goal_updates": self.goal_updates,
        }


class AdaptiveLearningProfile:
    """Isolates heuristic Q-tuning, failure tracking, and bias adjustment rules."""
    def __init__(self):
        self.state = {
            "global_transfer_bias": 1.0,
            "scenario_profiles": {},
            "route_profiles": {},
            "recent_rewards": [],
            "recent_failures": [],
            "blocked_routes": [],
            "blocked_route_ttl": {},
            "route_consecutive_failures": {},
            "none_failure_count": 0,
            "none_block_steps": 0,
            "plan_patterns": {},
            "consolidated_insights": [],
            "llm_memory_rules": {},
            "q_table": [],
        }

    def update_signal(self, scenario_mode, action, reward_score, kpis=None):
        self.state.setdefault("recent_rewards", []).append(round(reward_score, 2))
        self.state["recent_rewards"] = self.state["recent_rewards"][-50:]
        self._decay_blocked_routes()
        self._decay_none_block()
        
        failure_window = self.state.setdefault("recent_failures", [])
        if kpis:
            failure_window.append({
                "scenario": scenario_mode,
                "search_time": round(kpis.get("estimated_search_time_min", 0.0), 2),
                "queue_length": int(kpis.get("queue_length", 0)),
                "resilience_score": round(kpis.get("resilience_score", 0.0), 2),
                "action": (action or {}).get("action", "none"),
                "failed": bool(reward_score < 0 or kpis.get("estimated_search_time_min", 0.0) > 4.8 or kpis.get("queue_length", 0) >= 4),
            })
            self.state["recent_failures"] = failure_window[-30:]

        action_type = (action or {}).get("action", "none")
        queue_length = int((kpis or {}).get("queue_length", 0))
        none_failed = action_type == "none" and (reward_score < 0 or queue_length >= 4)
        if none_failed:
            self.state["none_failure_count"] = int(self.state.get("none_failure_count", 0)) + 1
            if self.state["none_failure_count"] >= 2:
                self.state["none_block_steps"] = max(int(self.state.get("none_block_steps", 0)), 4)
                self.state["latest_learning_insight"] = (
                    "Memory: repeated NONE outcomes failed under pressure; blocking hold and forcing a micro redirect."
                )
        elif action_type == "redirect" and reward_score > -0.1:
            self.state["none_failure_count"] = 0

        reward_window = self.state["recent_rewards"]
        if reward_window:
            avg_reward = sum(reward_window) / len(reward_window)
            current_bias = float(self.state.get("global_transfer_bias", 1.0))
            # Stronger reward-to-policy coupling — each update moves bias proportionally
            if avg_reward < -0.45:
                delta = -0.40
            elif avg_reward < -0.3:
                delta = -0.25   # aggressive downshift on bad streaks
            elif avg_reward < 0:
                delta = -0.10
            elif avg_reward > 0.3:
                delta = +0.15
            else:
                delta = +0.05
            self.state["global_transfer_bias"] = round(min(2.0, max(0.05, current_bias + delta)), 2)
            self.state["last_policy_reward"] = round(avg_reward, 3)

        scenario_profiles = self.state.setdefault("scenario_profiles", {})
        scenario_key = scenario_mode or "Unknown"
        scenario_profile = scenario_profiles.setdefault(
            scenario_key,
            {"steps": 0, "avg_reward": 0.0, "avg_search_time": 0.0, "avg_allocation_success": 0.0, "preferred_transfer_bias": 1.0},
        )
        scenario_profile["steps"] += 1
        steps = scenario_profile["steps"]
        scenario_profile["avg_reward"] = round(((scenario_profile["avg_reward"] * (steps - 1)) + reward_score) / steps, 2)
        if kpis:
            scenario_profile["avg_search_time"] = round(((scenario_profile["avg_search_time"] * (steps - 1)) + kpis.get("estimated_search_time_min", 0.0)) / steps, 2)
            scenario_profile["avg_allocation_success"] = round(((scenario_profile["avg_allocation_success"] * (steps - 1)) + kpis.get("allocation_success_pct", 0.0)) / steps, 2)
        
        scenario_bias = scenario_profile.get("preferred_transfer_bias", 1.0)
        scenario_profile["preferred_transfer_bias"] = round(min(1.5, max(0.5, scenario_bias + (0.08 if reward_score > 0 else -0.08))), 2)

        if action and action.get("action") == "redirect":
            route_profiles = self.state.setdefault("route_profiles", {})
            route_key = f"{action.get('from')}->{action.get('to')}"
            route_profile = route_profiles.setdefault(route_key, {"attempts": 0, "avg_reward": 0.0, "success_bias": 1.0})
            route_profile["attempts"] += 1
            attempts = route_profile["attempts"]
            route_profile["avg_reward"] = round(((route_profile["avg_reward"] * (attempts - 1)) + reward_score) / attempts, 2)
            if reward_score < -0.1:
                bias_delta = -0.18
            elif reward_score > 0.1:
                bias_delta = 0.12
            else:
                bias_delta = 0.04 if reward_score > 0 else -0.08
            route_profile["success_bias"] = round(min(1.6, max(0.2, route_profile["success_bias"] + bias_delta)), 2)
            
            if reward_score < -0.1:
                self.state["global_transfer_bias"] = round(max(0.1, float(self.state.get("global_transfer_bias", 1.0)) - 0.18), 2)
                self.state["latest_learning_insight"] = (
                    f"Learning: reward {reward_score} reduced transfer confidence for {route_key}; "
                    "next redirect will be smaller unless conditions improve."
                )

            if reward_score < -0.25:
                # Strong reward shift - dramatically alter global bounds
                self.state["global_transfer_bias"] = round(max(0.1, float(self.state.get("global_transfer_bias", 1.0)) - 0.5), 2)
                self.state["latest_learning_insight"] = f"CRITICAL SHIFT: System performance degraded (Reward: {reward_score}). Slashing global transfer bias to {self.state['global_transfer_bias']}x to force policy change."
                self.state["force_recovery_redirect_next_step"] = True
                if route_profile["avg_reward"] <= -0.2 or self.state.get("last_policy_reward", 0.0) <= -0.2:
                    self.add_failure(action.get("from"), action.get("to"), reason=f"Reward collapse ({reward_score}) triggered adaptive route penalty.")
            elif reward_score >= -0.1:
                feedback_direction = "reduced" if reward_score > 0 else "increased"
                self.state["latest_learning_insight"] = f"Previous execution on {route_key} {feedback_direction} search time. Strategy confidence is now {route_profile['success_bias']}x."
            # Trigger consolidation every 5 updates
            if attempts % 5 == 0:
                self.consolidate_patterns()

    def add_failure(self, from_zone, to_zone, reason="Manual intervention"):
        failure_log = self.state.setdefault("recent_failures", [])
        route_key = f"{from_zone}->{to_zone}"
        failure_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "from": from_zone,
            "to": to_zone,
            "route_key": route_key,
            "reason": reason,
            "failed": True
        })
        self.state["recent_failures"] = failure_log[-30:]

        # Track consecutive failures per route
        consec = self.state.setdefault("route_consecutive_failures", {})
        consec[route_key] = consec.get(route_key, 0) + 1

        # Auto-block routes after repeated failures, then let decay re-open them.
        blocked = self.state.setdefault("blocked_routes", [])
        blocked_ttl = self.state.setdefault("blocked_route_ttl", {})
        severe_negative = "negative reward" in str(reason).lower() or "reward collapse" in str(reason).lower()
        if (consec[route_key] >= 2 or severe_negative) and route_key not in blocked:
            blocked.append(route_key)
            blocked_ttl[route_key] = 6 if consec[route_key] >= 2 else 4
            self.state["latest_learning_insight"] = (
                f"Memory: Route {route_key} BLOCKED after {consec[route_key]} negative outcome(s) — "
                f"system will select alternative destinations."
            )
        # Decay: track successful steps per route to auto-unblock
        decay = self.state.setdefault("route_success_since_block", {})
        decay[route_key] = 0  # reset success counter on new failure

    def reset_route_failure_count(self, from_zone, to_zone):
        route_key = f"{from_zone}->{to_zone}"
        self.state.get("route_consecutive_failures", {})[route_key] = 0
        # Increment success counter: auto-unblock after 10 consecutive successes
        decay = self.state.setdefault("route_success_since_block", {})
        decay[route_key] = decay.get(route_key, 0) + 1
        blocked = self.state.get("blocked_routes", [])
        if route_key in blocked and decay[route_key] >= 4:
            blocked.remove(route_key)
            self.state.get("blocked_route_ttl", {}).pop(route_key, None)
            decay[route_key] = 0
            self.state["latest_learning_insight"] = (
                f"Memory: Route {route_key} AUTO-UNBLOCKED after recovery checks — restored to candidate pool."
            )

    def get_route_failure_count(self, from_zone, to_zone):
        route_key = f"{from_zone}->{to_zone}"
        return self.state.get("route_consecutive_failures", {}).get(route_key, 0)

    def _decay_blocked_routes(self):
        blocked = self.state.setdefault("blocked_routes", [])
        ttl = self.state.setdefault("blocked_route_ttl", {})
        for route in list(blocked):
            ttl[route] = int(ttl.get(route, 6)) - 1
            if ttl[route] <= 0:
                blocked.remove(route)
                ttl.pop(route, None)
                self.state.get("route_consecutive_failures", {})[route] = 0
                self.state["latest_learning_insight"] = (
                    f"Memory: Route {route} cooled down and is available for cautious retry."
                )

    def _decay_none_block(self):
        remaining = int(self.state.get("none_block_steps", 0) or 0)
        if remaining <= 0:
            return
        self.state["none_block_steps"] = remaining - 1
        if self.state["none_block_steps"] <= 0:
            self.state["none_failure_count"] = 0
            self.state["latest_learning_insight"] = "Memory: NONE hold block cooled down; cautious holds are available again."

    def load_from_payload(self, state):
        self.state = deepcopy(state)

    def export(self):
        return deepcopy(self.state)

    def get_public_profile(self, scenario_mode=None, from_zone=None, to_zone=None):
        scenario_profiles = self.state.get("scenario_profiles", {})
        route_profiles = self.state.get("route_profiles", {})
        route_key = f"{from_zone}->{to_zone}" if from_zone and to_zone else None
        blocked_routes = self.state.get("blocked_routes", [])
        consec_failures = self.state.get("route_consecutive_failures", {})

        # Build human-readable avoid hint for the planner
        avoid_hints = []
        for r in blocked_routes:
            count = consec_failures.get(r, 0)
            avoid_hints.append(f"{r} (blocked: {count} failures)")

        llm_decisions = self.state.get("llm_decisions", [])
        llm_rules = self._get_active_llm_rules(scenario_mode=scenario_mode, from_zone=from_zone, to_zone=to_zone)
        return {
            "global_transfer_bias": round(float(self.state.get("global_transfer_bias", 1.0)), 2),
            "recent_reward_avg": round(sum(self.state.get("recent_rewards", []) or [0.0]) / max(1, len(self.state.get("recent_rewards", []))), 2),
            "last_reward": (self.state.get("recent_rewards", []) or [-0.1])[-1],
            "last_policy_reward": float(self.state.get("last_policy_reward", 0.0)),
            "scenario_profile": deepcopy(scenario_profiles.get(scenario_mode, {})) if scenario_mode else {},
            "route_profile": deepcopy(route_profiles.get(route_key, {})) if route_key else {},
            "recent_failures": deepcopy(self.state.get("recent_failures", [])[-5:]),
            "blocked_routes": blocked_routes[:],
            "blocked_route_ttl": deepcopy(self.state.get("blocked_route_ttl", {})),
            "route_consecutive_failures": deepcopy(consec_failures),
            "avoid_hints": avoid_hints,
            "consolidated_insights": self.state.get("consolidated_insights", []),
            "latest_learning_insight": self.state.get("latest_learning_insight", "No specific route patterns consolidated yet."),
            "llm_decisions_count": len(llm_decisions),
            "last_llm_decision": deepcopy(llm_decisions[-1]) if llm_decisions else {},
            "llm_memory_rules": llm_rules,
            "force_hold_next_step": bool(self.state.get("force_hold_next_step", False)),
            "force_recovery_redirect_next_step": bool(self.state.get("force_recovery_redirect_next_step", False)),
            "none_failure_count": int(self.state.get("none_failure_count", 0) or 0),
            "none_block_steps": int(self.state.get("none_block_steps", 0) or 0),
            "none_block_active": int(self.state.get("none_block_steps", 0) or 0) > 0,
            "pattern_penalty_rules": dict(self.state.get("pattern_penalty_rules", {})),
            "plan_patterns": deepcopy(self.state.get("plan_patterns", {})),
        }

    def record_llm_rule(self, scenario_mode, planner_output, final_action, reward_score=0.0, kpis=None):
        planner_output = planner_output or {}
        final_action = final_action or planner_output.get("proposed_action", {})
        if final_action.get("action") != "redirect":
            return
        source = final_action.get("from")
        destination = final_action.get("to")
        if not source or not destination or source == destination:
            return

        rationale = (
            planner_output.get("llm_summary")
            or planner_output.get("rationale")
            or final_action.get("reason")
            or ""
        )
        scenario = scenario_mode or "Unknown"
        route_key = f"{source}->{destination}"
        rules = self.state.setdefault("llm_memory_rules", {})
        rule_key = f"{scenario}|{source}|{destination}"
        rule = rules.setdefault(
            rule_key,
            {
                "scenario": scenario,
                "from": source,
                "to": destination,
                "route_key": route_key,
                "prefer_count": 0,
                "avoid_count": 0,
                "strength": 0.0,
                "avg_reward": 0.0,
                "ttl": 10,
                "last_reason": "",
                "last_step": None,
                "source": "llm",
            },
        )

        reward = float(reward_score or 0.0)
        attempts = int(rule.get("prefer_count", 0)) + int(rule.get("avoid_count", 0)) + 1
        rule["avg_reward"] = round(((float(rule.get("avg_reward", 0.0)) * (attempts - 1)) + reward) / attempts, 3)
        negative_language = any(token in rationale.lower() for token in ["avoid", "fail", "blocked", "unsafe", "risk"])
        if reward < -0.2 or negative_language:
            rule["avoid_count"] = int(rule.get("avoid_count", 0)) + 1
            rule["strength"] = round(max(-1.0, float(rule.get("strength", 0.0)) - 0.25), 3)
        else:
            rule["prefer_count"] = int(rule.get("prefer_count", 0)) + 1
            bonus = 0.25 if planner_output.get("llm_influence") else 0.12
            rule["strength"] = round(min(1.0, float(rule.get("strength", 0.0)) + bonus + max(0.0, reward) * 0.1), 3)
        rule["ttl"] = 12
        rule["last_reason"] = rationale[:220]
        rule["last_step"] = planner_output.get("step")
        rule["source"] = planner_output.get("llm_source", "llm")
        rules[rule_key] = rule
        self.state["llm_memory_rules"] = dict(list(rules.items())[-80:])
        direction = "prefer" if rule["strength"] >= 0 else "avoid"
        self.state["latest_learning_insight"] = (
            f"LLM memory rule learned: {direction} {route_key} during {scenario} "
            f"(strength {rule['strength']})."
        )

    def _get_active_llm_rules(self, scenario_mode=None, from_zone=None, to_zone=None):
        rules = self.state.setdefault("llm_memory_rules", {})
        active = []
        for key, rule in list(rules.items()):
            ttl = int(rule.get("ttl", 0) or 0)
            if ttl <= 0:
                del rules[key]
                continue
            if scenario_mode and rule.get("scenario") not in {scenario_mode, "Unknown"}:
                continue
            if from_zone and rule.get("from") != from_zone:
                continue
            if to_zone and rule.get("to") != to_zone:
                continue
            active.append(deepcopy(rule))
        active.sort(key=lambda item: abs(float(item.get("strength", 0.0))), reverse=True)
        return active[:12]

    def record_plan_outcome(self, plan, critic, execution, reward_score):
        action = (execution or {}).get("final_action") or (critic or {}).get("revised_action") or (plan or {}).get("proposed_action", {})
        if action.get("action") != "redirect":
            signature = "NONE"
        else:
            volume = "micro" if int(action.get("vehicles", 0) or 0) <= 2 else "standard"
            signature = f"{action.get('from')}->{action.get('to')}:{volume}"
        patterns = self.state.setdefault("plan_patterns", {})
        profile = patterns.setdefault(
            signature,
            {
                "attempts": 0,
                "avg_reward": 0.0,
                "successes": 0,
                "failures": 0,
                "last_action": {},
                "last_critic": {},
            },
        )
        profile["attempts"] += 1
        attempts = profile["attempts"]
        profile["avg_reward"] = round(((profile["avg_reward"] * (attempts - 1)) + reward_score) / attempts, 3)
        if reward_score > 0:
            profile["successes"] += 1
        else:
            profile["failures"] += 1
            if profile["failures"] >= 2 and signature != "NONE":
                self.state["latest_learning_insight"] = (
                    f"Plan pattern {signature} is underperforming; future route scoring will penalize repeat micro-actions."
                )
        profile["last_action"] = deepcopy(action)
        profile["last_critic"] = {
            "approved": bool((critic or {}).get("approved")),
            "risk_score": (critic or {}).get("risk_score"),
            "risk_level": (critic or {}).get("risk_level"),
        }
        self.state["plan_patterns"] = dict(sorted(patterns.items())[-40:])

    def get_recently_failed_zones(self):
        """Returns a list of destination zones that failed within the last 5 attempts."""
        failures = self.state.get("recent_failures", [])[-5:]
        return list(set(f["to"] for f in failures if f.get("failed")))

    def set_q_table(self, q_table):
        self.state["q_table"] = deepcopy(q_table)

    def get_q_table(self):
        return deepcopy(self.state.get("q_table", []))

    def consolidate_patterns(self):
        """Discovers high-level operational failure patterns across history."""
        failures = self.state.get("recent_failures", [])
        if len(failures) < 5:
            return

        route_failure_stats = {}
        scenario_failure_stats = {}
        
        for f in failures:
            if not f.get("failed"): continue
            
            route_key = f.get("route_key") or f"{f.get('from')}->{f.get('to')}"
            scen = f.get("scenario", "Unknown")
            
            route_failure_stats[route_key] = route_failure_stats.get(route_key, 0) + 1
            scenario_failure_stats[scen] = scenario_failure_stats.get(scen, 0) + 1
            
        insights = []
        # Pattern 1: Route-specific failure density
        for route, count in route_failure_stats.items():
            if count >= 3:
                insights.append(f"CRITICAL PATTERN: Redirection to {route.split('->')[-1]} has failed {count} times recently. Recommend avoidance.")
        
        # Pattern 2: Scenario systemic instability
        for scen, count in scenario_failure_stats.items():
            if count >= 5:
                insights.append(f"SYSTEMIC PATTERN: {scen} scenario is showing high failure density ({count} events). System should prioritize safety over efficiency.")
                
        if insights:
            self.state["consolidated_insights"] = insights
            self.state["latest_learning_insight"] = insights[0]
        else:
            self.state["consolidated_insights"] = []

        # Logic-based rule generation: convert patterns into scoring penalties
        penalty_rules = self.state.setdefault("pattern_penalty_rules", {})
        for route, count in route_failure_stats.items():
            if count >= 3:
                # Assign a numeric penalty that the planner scoring can directly use
                penalty_rules[route] = round(min(0.5, 0.10 * count), 3)  # up to 50% penalty
        # Decay old rules not seen in recent failures
        for route in list(penalty_rules.keys()):
            if route not in route_failure_stats:
                penalty_rules[route] = round(max(0, penalty_rules[route] - 0.05), 3)
                if penalty_rules[route] <= 0:
                    del penalty_rules[route]

        for rule in self.state.setdefault("llm_memory_rules", {}).values():
            rule["ttl"] = max(0, int(rule.get("ttl", 0) or 0) - 1)


class AgentMemory:
    """Thread-safe orchestration layer for tracking persistent simulation structures."""
    def __init__(self, storage_path=None, max_history=200, max_cycles=200, flush_interval=10):
        base_dir = os.path.dirname(__file__)
        self.storage_path = storage_path or os.path.join(base_dir, "memory", "agent_memory_store.json")
        self.max_history = max_history
        self.max_cycles = max_cycles
        self.flush_interval = flush_interval
        
        # Modules
        self.metrics = MemoryMetricsEngine()
        self.learning = AdaptiveLearningProfile()
        
        # Concurrency Safety
        self._lock = threading.RLock()
        self._write_count = 0
        
        # Data
        self.schema_version = "2.0"
        self.history = []
        self.history_archives = [] # Compressed blocks
        self.cycles = []
        self.goal_history = []
        self.active_goal = {}
        
        self._load()

    def add(self, state, transition=None, summary=None, step=None, kpis=None, notifications=None, event_context=None):
        """Thread-safe structural addition mapped to O(1) engine triggers."""
        # Simple Schema validation
        if not isinstance(state, dict):
            logging.warning("AgentMemory: Expected state dict, discarding invalid append.")
            return

        with self._lock:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "step": step if step is not None else len(self.history) + 1,
                "state": deepcopy(state),
                "summary": deepcopy(summary) if summary else {},
                "transition": deepcopy(transition) if transition else {},
                "kpis": deepcopy(kpis) if kpis else {},
                "notifications": deepcopy(notifications) if notifications else [],
                "event_context": deepcopy(event_context) if event_context else {},
            }
            self.history.append(entry)
            
            # Map into O(1) performance engine
            free_slots_total = sum(zone.get("free_slots", 0) for zone in state.values())
            is_congested = any(zone.get("free_slots", 0) < 10 for zone in state.values())
            
            self.metrics.add_step(free_slots_total, is_congested)
            self.metrics.add_kpis(kpis)
            
            self._compress_history()
            self._throttled_save()

    def log_cycle(self, cycle_data):
        if not isinstance(cycle_data, dict): return
        
        with self._lock:
            self.cycles.append({
                "timestamp": datetime.utcnow().isoformat(),
                **deepcopy(cycle_data),
            })
            
            if "reward" in cycle_data and "agentic_reward_score" in cycle_data["reward"]:
                self.metrics.add_reward(cycle_data["reward"]["agentic_reward_score"])
                
            if cycle_data.get("planner_output", {}).get("llm_influence") or cycle_data.get("planner_output", {}).get("llm_advisory_used"):
                llm_mem = self.learning.state.setdefault("llm_decisions", [])
                llm_mem.append({
                    "step": cycle_data.get("step"),
                    "action": cycle_data["planner_output"].get("proposed_action"),
                    "reason": cycle_data["planner_output"].get("llm_summary", ""),
                    "influenced": bool(cycle_data.get("planner_output", {}).get("llm_influence")),
                    "source": cycle_data.get("planner_output", {}).get("llm_source", "llm"),
                })
                self.learning.state["llm_decisions"] = llm_mem[-50:]
            if len(self.cycles) > self.max_cycles:
                self.cycles = self.cycles[-self.max_cycles :]
            self._throttled_save()

    def set_goal(self, goal):
        with self._lock:
            comparable_current = {k: v for k, v in self.active_goal.items() if k != "timestamp"}
            if comparable_current == goal: return

            goal_record = {"timestamp": datetime.utcnow().isoformat(), **deepcopy(goal)}
            self.active_goal = goal_record
            self.goal_history.append(goal_record)
            self.goal_history = self.goal_history[-50:]
            self.metrics.goal_updates += 1
            self._throttled_save()

    def get_active_goal(self):
        with self._lock:
            return deepcopy(self.active_goal)

    def get_recent_states(self, limit=5):
        with self._lock:
            return deepcopy(self.history[-limit:])

    def get_recent_cycles(self, limit=5):
        with self._lock:
            return deepcopy(self.cycles[-limit:])

    def update_learning_signal(self, scenario_mode, action, reward_score, kpis=None):
        with self._lock:
            self.learning.update_signal(scenario_mode, action, reward_score, kpis)
            self._throttled_save()

    def record_plan_outcome(self, plan, critic, execution, reward_score):
        with self._lock:
            self.learning.record_plan_outcome(plan, critic, execution, reward_score)
            self._throttled_save()

    def record_llm_rule(self, scenario_mode, planner_output, final_action, reward_score=0.0, kpis=None):
        with self._lock:
            self.learning.record_llm_rule(scenario_mode, planner_output, final_action, reward_score, kpis)
            self._throttled_save()

    def add_failure(self, from_zone, to_zone, reason):
        with self._lock:
            self.learning.add_failure(from_zone, to_zone, reason)
            self._throttled_save()

    def set_q_table(self, q_table):
        with self._lock:
            self.learning.set_q_table(q_table)
            self._throttled_save()

    def get_q_table(self):
        with self._lock:
            return self.learning.get_q_table()

    def get_learning_profile(self, scenario_mode=None, from_zone=None, to_zone=None):
        with self._lock:
            return self.learning.get_public_profile(scenario_mode, from_zone, to_zone)

    def get_route_failure_count(self, from_zone, to_zone):
        with self._lock:
            return self.learning.get_route_failure_count(from_zone, to_zone)

    def reset_route_failure_count(self, from_zone, to_zone):
        with self._lock:
            self.learning.reset_route_failure_count(from_zone, to_zone)
            self._throttled_save()

    def get_metrics(self):
        with self._lock:
            payload = self.metrics.get_aggregated_metrics()
            payload["active_goal"] = deepcopy(self.active_goal)
            payload["learning_profile"] = self.learning.get_public_profile()
            return payload

    def export(self):
        with self._lock:
            return {
                "_version": self.schema_version,
                "history": deepcopy(self.history),
                "history_archives": deepcopy(self.history_archives),
                "cycles": deepcopy(self.cycles),
                "goal_history": deepcopy(self.goal_history),
                "active_goal": deepcopy(self.active_goal),
                "learning_state": self.learning.export(),
                "metrics_state": self.metrics.export(),
            }

    def load_export(self, payload):
        with self._lock:
            self.history = deepcopy(payload.get("history", []))
            self.history_archives = deepcopy(payload.get("history_archives", []))
            self.cycles = deepcopy(payload.get("cycles", []))
            self.goal_history = deepcopy(payload.get("goal_history", []))
            self.active_goal = deepcopy(payload.get("active_goal", {}))
            
            # Use backward compatible fallback extraction mapping natively
            self.learning.load_from_payload(payload.get("learning_state", self.learning.export()))
            if "metrics_state" in payload:
                self.metrics.load_from_payload(payload["metrics_state"])
            
            self.flush()

    def reset(self, persist=True):
        with self._lock:
            self.history = []
            self.history_archives = []
            self.cycles = []
            self.goal_history = []
            self.active_goal = {}
            self.learning = AdaptiveLearningProfile()
            self.metrics = MemoryMetricsEngine()
            if persist:
                self.flush()

    def _compress_history(self):
        """Compresses old memory chunks into space-saving archives preventing JSON bloat."""
        if len(self.history) > self.max_history:
            chunk = self.history[: self.max_history // 4]
            self.history = self.history[self.max_history // 4 :]
            
            # Create a mathematical rollup archive block instead of dropping
            if chunk:
                self.history_archives.append({
                    "start_step": chunk[0].get("step"),
                    "end_step": chunk[-1].get("step"),
                    "archived_at": datetime.utcnow().isoformat(),
                    "compressed_size": len(chunk)
                })
                # Cap archives 
                self.history_archives = self.history_archives[-50:]

    def _throttled_save(self):
        """Reduces block I/O operations mathematically."""
        self._write_count += 1
        if self._write_count % self.flush_interval == 0:
            self._save_to_disk()

    def flush(self):
        """Force writes immediately."""
        with self._lock:
            self._write_count = 0
            self._save_to_disk()

    def _load(self):
        if not os.path.exists(self.storage_path): return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
                
                # Check version migrations safely
                if payload.get("_version") != self.schema_version:
                    logging.info(f"AgentMemory mapping legacy migration from {payload.get('_version', '1.0')} -> {self.schema_version}")
                
                self.load_export(payload)
                
        except json.JSONDecodeError as err:
            logging.error(f"AgentMemory payload corruption: JSON decode failed -> {err}")
        except OSError as err:
            logging.error(f"AgentMemory IO Read Error -> {err}")
        except Exception as err:
            logging.error(f"AgentMemory unknown loading error structure -> {err}")

    def _save_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as file:
                json.dump(self.export(), file, indent=2)
        except OSError as e:
            logging.error(f"AgentMemory IO Save Error: -> {e}")
