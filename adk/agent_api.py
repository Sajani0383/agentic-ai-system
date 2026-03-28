from fastapi import FastAPI
from pydantic import BaseModel

from adk.agent_manager import run_agent, get_trace


class RunRequest(BaseModel):
    input: str


app = FastAPI(
    title="Smart Parking Agent API",
    description="API for querying the smart parking agent and reading its trace history.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "Smart Parking Agent API",
        "version": "1.0.0",
        "health": "/health",
        "docs": "/docs",
        "run_endpoint": "/run",
        "trace_endpoint": "/trace",
    }


@app.post("/run")
def run(query: RunRequest):
    result = run_agent(query.input)
    return {"response": result}


@app.get("/trace")
def trace():
    return {"trace": get_trace()}


@app.get("/health")
def health():
    return {"status": "ok"}
