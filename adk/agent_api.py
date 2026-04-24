import logging
import os
import time
from threading import Event, Lock, Thread
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from adk.agent_manager import (
    get_notification_feed,
    get_runtime_snapshot,
    get_trace,
    reset_runtime,
    run_agent,
    set_runtime_scenario,
    step_runtime,
)
from services.parking_runtime import runtime_service

logger = logging.getLogger("smart_parking.agent_api")
logger.addHandler(logging.NullHandler())
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


class AgentAPIError(RuntimeError):
    pass


class RunRequest(BaseModel):
    input: str


class ResetRequest(BaseModel):
    clear_memory: bool = False


class ScenarioRequest(BaseModel):
    scenario_mode: str


class BenchmarkRequest(BaseModel):
    episodes: int = 3
    steps_per_episode: int = 10


class AutonomyRequest(BaseModel):
    interval_seconds: float = 2.0
    max_steps: Optional[int] = None


class AutonomousLoop:
    def __init__(self, runtime):
        self.runtime = runtime
        self._stop_event = Event()
        self._thread = None
        self._lock = Lock()
        self._last_error = ""
        self._last_step = None
        self._started_at = None
        self._completed_steps = 0
        self._interval_seconds = 0.0
        self._max_steps = None

    def start(self, interval_seconds=2.0, max_steps=None):
        with self._lock:
            if self.is_running():
                raise AgentAPIError("Autonomous loop is already running.")
            self._stop_event.clear()
            self._last_error = ""
            self._last_step = None
            self._started_at = time.time()
            self._completed_steps = 0
            self._interval_seconds = interval_seconds
            self._max_steps = max_steps
            self._thread = Thread(
                target=self._run,
                args=(interval_seconds, max_steps),
                daemon=True,
                name="parking-autonomy-loop",
            )
            self._thread.start()
            logger.info("Autonomous loop started interval=%s max_steps=%s", interval_seconds, max_steps)
            return self.status()

    def stop(self):
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3)
        logger.info("Autonomous loop stopped")
        return self.status()

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def status(self):
        return {
            "running": self.is_running(),
            "started_at_epoch": self._started_at,
            "interval_seconds": self._interval_seconds,
            "max_steps": self._max_steps,
            "completed_steps": self._completed_steps,
            "last_step": self._last_step,
            "last_error": self._last_error,
        }

    def _run(self, interval_seconds, max_steps):
        while not self._stop_event.is_set():
            if max_steps is not None and self._completed_steps >= max_steps:
                break
            try:
                result = self.runtime.step()
                self._last_step = {
                    "step_number": result.get("step_number"),
                    "mode": result.get("mode"),
                    "action": result.get("action"),
                    "reward_score": result.get("reward_score"),
                    "kpis": result.get("kpis", {}),
                }
                self._completed_steps += 1
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Autonomous loop step failed")
            self._stop_event.wait(interval_seconds)


autonomous_loop = AutonomousLoop(runtime_service)


app = FastAPI(
    title="SRM Smart Parking Agent API",
    description="API for the SRM agentic parking runtime, goals, traces, and control loop.",
    version="2.0.0",
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected_key = os.getenv("PARKING_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return True


def safe_call(operation, error_message="Request failed"):
    try:
        return operation()
    except HTTPException:
        raise
    except AgentAPIError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("%s: %s", error_message, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{error_message}: {type(exc).__name__}",
        ) from exc


def _validate_non_empty(value, field_name):
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} cannot be empty.",
        )


def _valid_scenarios():
    return list(runtime_service.environment.event_catalog.keys())


def _build_learning_report(snapshot):
    metrics = snapshot.get("metrics", {})
    learning_profile = metrics.get("learning_profile", {})
    q_table = learning_profile.get("q_table", [])
    q_rows = len(q_table)
    q_cols = len(q_table[0]) if q_rows and isinstance(q_table[0], list) else 0
    return {
        "summary": {
            "steps": metrics.get("steps", 0),
            "avg_reward_score": metrics.get("avg_reward_score", 0.0),
            "goal_updates": metrics.get("goal_updates", 0),
            "q_table_shape": [q_rows, q_cols],
            "recent_reward_avg": metrics.get("learning_profile", {}).get("recent_reward_avg", 0.0),
        },
        "reward_updates": learning_profile.get("recent_rewards", []),
        "scenario_profiles": learning_profile.get("scenario_profiles", {}),
        "route_profiles": learning_profile.get("route_profiles", {}),
        "recent_failures": learning_profile.get("recent_failures", []),
        "q_table": q_table,
    }


def _build_metrics_report(snapshot):
    latest_result = snapshot.get("latest_result", {})
    return {
        "operational_kpis": snapshot.get("kpis", {}),
        "memory_metrics": snapshot.get("metrics", {}),
        "latest_action": latest_result.get("action", {}),
        "latest_policy_action": latest_result.get("policy_action", {}),
        "latest_reward": {
            "environment_reward": latest_result.get("environment_reward"),
            "reward_score": latest_result.get("reward_score"),
        },
        "autonomy": latest_result.get("autonomy", {}),
        "benchmark": snapshot.get("benchmark", {}),
    }


@app.get("/")
def root(_authorized=Depends(require_api_key)):
    return {
        "name": "SRM Smart Parking Agent API",
        "version": "2.0.0",
        "health": "/health",
        "docs": "/docs",
        "run_endpoint": "/run",
        "step_endpoint": "/step",
        "state_endpoint": "/state",
        "trace_endpoint": "/trace",
        "notifications_endpoint": "/notifications",
        "reset_endpoint": "/reset",
        "scenario_endpoint": "/scenario",
        "benchmark_endpoint": "/benchmark",
        "metrics_endpoint": "/metrics",
        "learning_endpoint": "/learning",
        "decision_endpoint": "/decision",
        "agents_endpoint": "/agents",
        "autonomy_endpoint": "/autonomy/status",
        "capabilities_endpoint": "/capabilities",
        "visualization_endpoint": "/visualization",
    }


@app.post("/run")
def run(query: RunRequest, _authorized=Depends(require_api_key)):
    _validate_non_empty(query.input, "input")
    return safe_call(lambda: run_agent(query.input), "Agent command failed")


@app.post("/step")
def step(_authorized=Depends(require_api_key)):
    return safe_call(step_runtime, "Agent step failed")


@app.post("/reset")
def reset(request: ResetRequest, _authorized=Depends(require_api_key)):
    autonomous_loop.stop()
    return safe_call(lambda: reset_runtime(clear_memory=request.clear_memory), "Runtime reset failed")


@app.post("/scenario")
def scenario(request: ScenarioRequest, _authorized=Depends(require_api_key)):
    _validate_non_empty(request.scenario_mode, "scenario_mode")
    valid_scenarios = _valid_scenarios()
    if request.scenario_mode not in valid_scenarios:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Unknown scenario_mode.",
                "valid_scenarios": valid_scenarios,
            },
        )
    return safe_call(lambda: set_runtime_scenario(request.scenario_mode), "Scenario update failed")


@app.post("/benchmark")
def benchmark(request: BenchmarkRequest, _authorized=Depends(require_api_key)):
    if not 1 <= request.episodes <= 20:
        raise HTTPException(status_code=422, detail="episodes must be between 1 and 20.")
    if not 1 <= request.steps_per_episode <= 100:
        raise HTTPException(status_code=422, detail="steps_per_episode must be between 1 and 100.")

    result = safe_call(
        lambda: runtime_service.run_benchmark(
            episodes=request.episodes,
            steps_per_episode=request.steps_per_episode,
        ),
        "Benchmark failed",
    )
    return {
        **result,
        "evaluation_explanation": {
            "agentic": "Full planner, critic, executor, reward, and Q-learning policy loop.",
            "baseline": "Environment simulation with no redirect action.",
            "delta_search_time": "Positive value means average minutes saved by the agentic loop.",
            "delta_resilience": "Positive value means stronger simulated disruption handling.",
            "delta_hotspots": "Positive value means fewer congestion hotspots.",
        },
    }


@app.post("/autonomy/start")
def start_autonomy(request: AutonomyRequest, _authorized=Depends(require_api_key)):
    if not 0.5 <= request.interval_seconds <= 60:
        raise HTTPException(status_code=422, detail="interval_seconds must be between 0.5 and 60.")
    if request.max_steps is not None and not 1 <= request.max_steps <= 10000:
        raise HTTPException(status_code=422, detail="max_steps must be between 1 and 10000.")
    return safe_call(
        lambda: autonomous_loop.start(
            interval_seconds=request.interval_seconds,
            max_steps=request.max_steps,
        ),
        "Autonomous loop start failed",
    )


@app.post("/autonomy/stop")
def stop_autonomy(_authorized=Depends(require_api_key)):
    return safe_call(autonomous_loop.stop, "Autonomous loop stop failed")


@app.get("/autonomy/status")
def autonomy_status(_authorized=Depends(require_api_key)):
    return autonomous_loop.status()


@app.get("/state")
def state(_authorized=Depends(require_api_key)):
    return safe_call(get_runtime_snapshot, "State read failed")


@app.get("/trace")
def trace(_authorized=Depends(require_api_key)):
    return safe_call(get_trace, "Trace read failed")


@app.get("/notifications")
def notifications(_authorized=Depends(require_api_key)):
    return safe_call(lambda: {"deliveries": get_notification_feed()}, "Notification read failed")


@app.get("/metrics")
def metrics(_authorized=Depends(require_api_key)):
    return safe_call(lambda: _build_metrics_report(get_runtime_snapshot()), "Metrics read failed")


@app.get("/learning")
def learning(_authorized=Depends(require_api_key)):
    return safe_call(lambda: _build_learning_report(get_runtime_snapshot()), "Learning read failed")


@app.get("/decision")
def decision(_authorized=Depends(require_api_key)):
    def build():
        latest = get_runtime_snapshot().get("latest_result", {})
        return {
            "mode": latest.get("mode"),
            "planner_output": latest.get("planner_output", {}),
            "critic_output": latest.get("critic_output", {}),
            "execution_output": latest.get("execution_output", {}),
            "policy_action": latest.get("policy_action", {}),
            "final_action": latest.get("action", {}),
            "reward": {
                "environment_reward": latest.get("environment_reward"),
                "reward_score": latest.get("reward_score"),
            },
            "autonomy": latest.get("autonomy", {}),
        }

    return safe_call(build, "Decision read failed")


@app.get("/agents")
def agents(_authorized=Depends(require_api_key)):
    def build():
        latest = get_runtime_snapshot().get("latest_result", {})
        return {
            "agents": [
                "MonitoringAgent",
                "DemandAgent",
                "BayesianAgent",
                "PlannerAgent",
                "CriticAgent",
                "ExecutorAgent",
                "PolicyAgent",
                "RewardAgent",
            ],
            "latest_interactions": latest.get("agent_interactions", []),
        }

    return safe_call(build, "Agent interaction read failed")


@app.get("/scenarios")
def scenarios(_authorized=Depends(require_api_key)):
    return {
        "active": runtime_service.environment.get_scenario_mode(),
        "available": _valid_scenarios(),
        "catalog": runtime_service.environment.event_catalog,
    }


@app.get("/visualization")
def visualization(_authorized=Depends(require_api_key)):
    def build():
        snapshot = get_runtime_snapshot()
        state_payload = snapshot.get("state", {})
        return {
            "chart_data": [
                {
                    "zone": zone,
                    "occupied": data.get("occupied", 0),
                    "free_slots": data.get("free_slots", 0),
                    "total_slots": data.get("total_slots", 0),
                    "entry": data.get("entry", 0),
                    "exit": data.get("exit", 0),
                }
                for zone, data in state_payload.items()
            ],
            "kpi_cards": snapshot.get("kpis", {}),
            "recent_cycles": snapshot.get("recent_cycles", []),
        }

    return safe_call(build, "Visualization data read failed")


@app.get("/capabilities")
def capabilities(_authorized=Depends(require_api_key)):
    snapshot = get_runtime_snapshot()
    return {
        "implemented": {
            "agent_control_api": True,
            "background_autonomy_loop": autonomous_loop.is_running(),
            "planner_critic_executor_loop": True,
            "q_learning_policy": True,
            "reward_tracking": True,
            "structured_kpis": True,
            "benchmarking": True,
            "mock_notifications": True,
            "file_persistence": True,
            "dashboard_available": True,
            "optional_api_key_auth": bool(os.getenv("PARKING_API_KEY")),
        },
        "simulation_boundaries": {
            "real_sensor_integration": False,
            "database_storage": False,
            "multi_user_queueing": False,
            "production_auth_rbac": False,
        },
        "active_scenario": snapshot.get("scenario_mode"),
        "llm_status": snapshot.get("llm_status", {}),
    }


@app.get("/health")
def health(_authorized=Depends(require_api_key)):
    snapshot = safe_call(get_runtime_snapshot, "Health check failed")
    return {
        "status": "ok",
        "goal": snapshot.get("goal", {}),
        "steps_recorded": snapshot.get("metrics", {}).get("steps", 0),
        "autonomy": autonomous_loop.status(),
        "llm_available": snapshot.get("llm_status", {}).get("available", False),
    }
