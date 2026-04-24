import logging
from copy import deepcopy
from datetime import datetime

from services.parking_runtime import runtime_service

logger = logging.getLogger("smart_parking.agent_manager")
logger.addHandler(logging.NullHandler())

MANAGER_TRACE_LIMIT = 200
manager_trace = []
strategy_hooks = {}


class AgentManagerError(ValueError):
    pass


def _utc_now():
    return datetime.utcnow().isoformat()


def _record_trace(label, payload=None, status="ok"):
    entry = {
        "timestamp": _utc_now(),
        "layer": "agent_manager",
        "label": label,
        "status": status,
        "payload": deepcopy(payload) if payload else {},
    }
    manager_trace.append(entry)
    del manager_trace[:-MANAGER_TRACE_LIMIT]
    return entry


def _safe_call(label, operation, fallback=None):
    try:
        result = operation()
        _record_trace(label, _summarize_result(result))
        return result
    except AgentManagerError:
        raise
    except Exception as exc:
        error_payload = {"error": f"{type(exc).__name__}: {exc}"}
        logger.exception("Agent manager operation failed: %s", label)
        _record_trace(label, error_payload, status="error")
        if fallback is not None:
            return fallback
        raise AgentManagerError(error_payload["error"]) from exc


def _summarize_result(result):
    if not isinstance(result, dict):
        return {"result_type": type(result).__name__}
    return {
        "type": result.get("type"),
        "mode": result.get("mode"),
        "step_number": result.get("step_number"),
        "action": result.get("action"),
        "goal": result.get("goal"),
        "message": result.get("message"),
    }


def preprocess_user_input(user_input):
    if user_input is None:
        raise AgentManagerError("user_input is required.")
    normalized = str(user_input).strip()
    if not normalized:
        raise AgentManagerError("user_input cannot be empty.")
    return normalized


def get_agent_registry():
    return {
        "runtime_id": "default_parking_runtime",
        "coordination_model": "single runtime with coordinated internal agents",
        "agents": [
            {
                "id": "monitoring",
                "name": "MonitoringAgent",
                "role": "Observe parking state from the environment.",
            },
            {
                "id": "demand",
                "name": "DemandAgent",
                "role": "Predict SRM block-level demand pressure.",
            },
            {
                "id": "bayesian",
                "name": "BayesianAgent",
                "role": "Infer congestion risk and confidence.",
            },
            {
                "id": "planner",
                "name": "PlannerAgent",
                "role": "Build goal-aware plan using tools, memory, and optional Gemini reasoning.",
            },
            {
                "id": "critic",
                "name": "CriticAgent",
                "role": "Review the plan for risk, safety, and goal alignment.",
            },
            {
                "id": "executor",
                "name": "ExecutorAgent",
                "role": "Convert approved plan into environment action.",
            },
            {
                "id": "policy",
                "name": "PolicyAgent",
                "role": "Expose Q-learning informed policy baseline and update Q-values.",
            },
            {
                "id": "reward",
                "name": "RewardAgent",
                "role": "Score the transition and feed learning memory.",
            },
        ],
    }


def observe():
    return _safe_call("observe", runtime_service.environment.get_state)


def decide():
    def build_decision():
        snapshot = runtime_service.get_runtime_snapshot()
        latest = snapshot.get("latest_result", {})
        return {
            "mode": latest.get("mode"),
            "goal": latest.get("goal", snapshot.get("goal", {})),
            "planner_output": latest.get("planner_output", {}),
            "critic_output": latest.get("critic_output", {}),
            "execution_output": latest.get("execution_output", {}),
            "policy_action": latest.get("policy_action", {}),
            "final_action": latest.get("action", {}),
            "reasoning": latest.get("reasoning", ""),
        }

    return _safe_call("decide", build_decision)


def act():
    return _safe_call("observe_decide_act", runtime_service.step)


def run_agent_loop(steps=1):
    if not isinstance(steps, int) or steps < 1 or steps > 100:
        raise AgentManagerError("steps must be an integer between 1 and 100.")

    results = []
    for _ in range(steps):
        results.append(act())

    return {
        "loop": "Observe -> Plan -> Criticize -> Act -> Reward -> Learn",
        "steps": steps,
        "results": results,
    }


def run_agent(user_input):
    normalized = preprocess_user_input(user_input)
    result = _safe_call(
        "run_agent_command",
        lambda: runtime_service.run_agent_command(normalized),
        fallback={
            "type": "error",
            "message": "Agent manager could not execute the command.",
            "result": runtime_service.get_runtime_snapshot(),
        },
    )
    if isinstance(result, dict):
        result.setdefault("manager", {})
        result["manager"].update(
            {
                "validated_input": normalized,
                "orchestration_layer": "agent_manager",
                "agent_registry": get_agent_registry()["agents"],
            }
        )
    return result


def get_trace():
    snapshot = runtime_service.get_runtime_snapshot()
    return {
        "runtime_trace": snapshot.get("trace", []),
        "manager_trace": deepcopy(manager_trace[-50:]),
    }


def get_runtime_snapshot():
    snapshot = runtime_service.get_runtime_snapshot()
    snapshot["manager"] = {
        "agent_registry": get_agent_registry(),
        "strategy_hooks": get_strategy_hooks(),
        "trace_count": len(manager_trace),
    }
    return snapshot


def get_notification_feed():
    return _safe_call("notification_feed", runtime_service.get_notification_feed, fallback=[])


def step_runtime():
    return act()


def reset_runtime(clear_memory=False):
    result = _safe_call(
        "reset_runtime",
        lambda: runtime_service.reset(clear_memory=clear_memory),
    )
    _record_trace("memory_reset" if clear_memory else "runtime_reset", {"clear_memory": clear_memory})
    return result


def set_runtime_scenario(scenario_mode):
    if scenario_mode not in runtime_service.environment.event_catalog:
        raise AgentManagerError(f"Unknown scenario: {scenario_mode}")
    return _safe_call(
        "set_runtime_scenario",
        lambda: runtime_service.set_scenario_mode(scenario_mode),
    )


def set_goal(goal):
    if not isinstance(goal, dict):
        raise AgentManagerError("goal must be a dictionary.")
    objective = str(goal.get("objective", "")).strip()
    if not objective:
        raise AgentManagerError("goal.objective is required.")
    runtime_service.memory.set_goal(goal)
    _record_trace("set_goal", goal)
    return runtime_service.get_runtime_snapshot().get("goal", {})


def update_goal(**updates):
    current_goal = runtime_service.memory.get_active_goal()
    if not current_goal:
        current_goal = {
            "objective": "Reduce SRM block congestion and parking search time.",
            "target_congested_zones": 1,
            "horizon_steps": 5,
            "target_search_time_min": 4.0,
        }
    current_goal.update({key: value for key, value in updates.items() if value is not None})
    return set_goal(current_goal)


def get_memory_report(limit=10):
    snapshot = runtime_service.get_runtime_snapshot()
    return {
        "short_term_memory": snapshot.get("recent_cycles", [])[-limit:],
        "long_term_memory": snapshot.get("metrics", {}).get("learning_profile", {}),
        "recent_states": snapshot.get("recent_states", [])[-limit:],
        "active_goal": snapshot.get("goal", {}),
    }


def get_policy_learning_report():
    snapshot = runtime_service.get_runtime_snapshot()
    metrics = snapshot.get("metrics", {})
    learning_profile = metrics.get("learning_profile", {})
    q_table = learning_profile.get("q_table", [])
    return {
        "policy_type": "Q-learning informed SRM parking-block selection",
        "q_table": q_table,
        "q_table_shape": [
            len(q_table),
            len(q_table[0]) if q_table and isinstance(q_table[0], list) else 0,
        ],
        "recent_rewards": learning_profile.get("recent_rewards", []),
        "scenario_profiles": learning_profile.get("scenario_profiles", {}),
        "route_profiles": learning_profile.get("route_profiles", {}),
        "latest_policy_action": snapshot.get("latest_result", {}).get("policy_action", {}),
        "latest_reward": {
            "environment_reward": snapshot.get("latest_result", {}).get("environment_reward"),
            "reward_score": snapshot.get("latest_result", {}).get("reward_score"),
        },
    }


def register_strategy_hook(name, handler):
    if not name or not str(name).strip():
        raise AgentManagerError("strategy hook name is required.")
    if not callable(handler):
        raise AgentManagerError("strategy hook handler must be callable.")
    strategy_hooks[str(name).strip()] = handler
    _record_trace("register_strategy_hook", {"name": str(name).strip()})
    return get_strategy_hooks()


def get_strategy_hooks():
    return {
        "built_in": [
            "agentic_planner_critic_executor",
            "q_learning_policy_baseline",
            "local_fallback_reasoning",
            "optional_gemini_reasoning",
        ],
        "registered": sorted(strategy_hooks.keys()),
    }


def run_strategy_hook(name, payload=None):
    handler = strategy_hooks.get(name)
    if handler is None:
        raise AgentManagerError(f"Unknown strategy hook: {name}")
    return _safe_call(
        f"strategy_hook:{name}",
        lambda: handler(payload or {}, runtime_service),
    )
