from fastapi import FastAPI
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


class RunRequest(BaseModel):
    input: str


class ResetRequest(BaseModel):
    clear_memory: bool = False


class ScenarioRequest(BaseModel):
    scenario_mode: str


app = FastAPI(
    title="Smart Parking Agent API",
    description="API for the agentic parking runtime, goals, traces, and control loop.",
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "name": "Smart Parking Agent API",
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
    }


@app.post("/run")
def run(query: RunRequest):
    return run_agent(query.input)


@app.post("/step")
def step():
    return step_runtime()


@app.post("/reset")
def reset(request: ResetRequest):
    return reset_runtime(clear_memory=request.clear_memory)


@app.post("/scenario")
def scenario(request: ScenarioRequest):
    return set_runtime_scenario(request.scenario_mode)


@app.get("/state")
def state():
    return get_runtime_snapshot()


@app.get("/trace")
def trace():
    return {"trace": get_trace()}


@app.get("/notifications")
def notifications():
    return {"deliveries": get_notification_feed()}


@app.get("/health")
def health():
    snapshot = get_runtime_snapshot()
    return {
        "status": "ok",
        "goal": snapshot.get("goal", {}),
        "steps_recorded": snapshot.get("metrics", {}).get("steps", 0),
    }
