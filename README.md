# SRM Smart Parking Agentic AI System

A simulation-first agentic AI parking project designed for SRM campus parking management, with:

- SRM block-wise parking simulation
- planner / critic / executor agent loop
- strict Planner -> Critic -> Executor control hierarchy
- bounded re-planning and failed-execution recovery
- belief-state, route-risk, and reward-trend tool use
- lightweight multi-step action evaluation
- adaptive reward learning plus Q-learning policy updates
- plan-outcome memory for successful and failed route patterns
- demand estimation and congestion-aware routing across SRM parking blocks
- Streamlit command-center dashboard
- FastAPI runtime and mock notification API
- optional Gemini-backed reasoning when API keys are available

## Project Structure

- `simulation.py`: CLI simulation entry point
- `agent_controller.py`: main orchestration layer for agents and environment
- `environment/parking_environment.py`: SRM parking environment, block capacities, and reward logic
- `agents/`: monitoring, demand, Bayesian, planner, critic, executor, policy, and reward agents
- `ui/adk_dashboard.py`: Streamlit dashboard
- `adk/agent_api.py`: FastAPI app
- `ml/`: demand prediction model loading and training
- `services/mock_notification_service.py`: mock delivery feed for app, SMS, and signage outputs
- `tests/`: simulation validation tests
- `start_project.ps1`: start API and UI
- `stop_project.ps1`: stop API and UI

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

On macOS/Linux, use `python3`/`pip3` if `python` is not available:

```bash
pip3 install -r requirements.txt
```

Create your env file:

```powershell
Copy-Item .env.example .env
```

## Run

Start everything:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_project.ps1
```

Open:

- Dashboard: `http://127.0.0.1:8501`
- API health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`

## Stop

```powershell
powershell -ExecutionPolicy Bypass -File .\stop_project.ps1
```

## Manual Commands

Run the simulation:

```powershell
python simulation.py
```

Run the API:

```powershell
python -m uvicorn adk.agent_api:app --host 127.0.0.1 --port 8000
```

Run the dashboard:

```powershell
python -m streamlit run ui/adk_dashboard.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

## SRM Simulation-First Mode

This project is complete as an SRM parking simulation even without real hardware, live feeds, or external APIs.

- demand prediction falls back safely if the saved model cannot be loaded
- planner and critic continue through tool-driven local reasoning if no LLM key is available
- notifications are delivered through a mock API feed so the dashboard still demonstrates proactive action

## SRM Agentic AI Backend Formulation

The backend is formulated as a bounded multi-agent decision system for SRM campus parking control. Each simulation cycle follows a strict pipeline:

1. `MonitoringAgent` observes the current parking state.
2. `DemandAgent` predicts SRM block pressure and demand drift.
3. `BayesianAgent` estimates uncertainty and congestion risk.
4. `PlannerAgent` builds a goal-aligned action plan using tool observations, belief state, route risk, reward trend, blocked-route memory, and lightweight future-effect scoring.
5. `CriticAgent` validates safety, risk, utility, and hard constraints, then approves, reduces, rejects, or requests a bounded replan.
6. `ExecutorAgent` validates executable capacity and prepares the final action.
7. `RewardAgent` scores the outcome.
8. `AgentMemory` stores route failures, blocked routes with decay, repeated failed holds, plan patterns, reward trends, and Q-table updates.

The controller enforces these invariants:

- policy output is advisory only and cannot override a critic-approved planner decision
- an approved redirect must execute with `executed_vehicles > 0`, otherwise it is treated as a failed execution and recovery is attempted
- `executed_vehicles <= executable_vehicles`
- repeated failed `NONE` decisions are temporarily blocked
- repeated route failures create temporary hard route blocks
- low-confidence redirects are reduced to micro-actions
- pressure conditions force a safe 1-2 vehicle micro-redirect instead of repeated holding
- malformed planner, critic, or executor payloads are normalized by backend contracts before reaching the dashboard

This makes the project a hybrid agentic AI simulation for SRM parking: it has autonomous planning, critique, execution validation, learning memory, bounded replanning, tool-grounded belief state, and reward-driven adaptation. The deterministic guardrails are intentional agentic safety controls, not a fallback weakness: they keep the simulated control loop reliable when a cloud LLM is skipped, rate-limited, or unnecessary.

## SRM Parking Blocks

The simulation uses SRM parking block values as the default environment:

| Block | Car Slots | Bike Slots | Total Capacity |
|---|---:|---:|---:|
| Main Block | 120 | 300 | 420 |
| Hi-Tech Block | 150 | 400 | 550 |
| ES Block | 100 | 250 | 350 |
| Mech A | 80 | 200 | 280 |
| Mech B | 90 | 220 | 310 |
| Mech C | 85 | 210 | 295 |
| Automobile Block | 70 | 180 | 250 |
| CRC Block | 130 | 150 | 280 |
| Admin Block | 110 | 120 | 230 |
| Library | 60 | 200 | 260 |
| MBA Block | 75 | 160 | 235 |
| Biotech Block | 90 | 180 | 270 |
| Tech Park | 200 | 300 | 500 |
| Basic Eng Lab | 140 | 350 | 490 |

## Environment Model

The SRM parking environment is intentionally simulation-first. For viva or presentation, you can explain each step in five phases:

1. advance simulated time and load the current event profile
2. build operational signals such as weather, queues, reserved slots, and temporary block disruptions
3. estimate SRM block-level entry and exit flow from demand, time, and event pressure
4. reroute incoming arrivals between SRM blocks when the agents issue a redirect action
5. update occupancy, KPIs, notifications, and the transition report

The environment is now configurable through an internal config map, validates redirect actions and SRM block state, and exposes a simple environment summary for explanation-friendly walkthroughs.

## Optional LLM Mode

If you add a valid Gemini key in `.env`, the planner and critic can use live model reasoning:

```env
ENABLE_LLM=true
GOOGLE_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-flash-latest
```

Without a valid key, or when Gemini quota is exhausted, the project still runs as a fully functional simulation and records the fallback path in the agent decision table.

Check Gemini directly:

```bash
python3 -c "import llm_reasoning; llm=llm_reasoning.get_llm(); print('llm', bool(llm)); print(llm.invoke('Reply with exactly: ok').content if llm else 'no llm')"
```

If this prints `ok`, the Gemini key, model, and network path are working. If it returns `429 RESOURCE_EXHAUSTED`, Gemini is reachable but the active quota is exhausted, so the dashboard will show `Gemini attempted -> local fallback used`.

## Tests

Run the simulation validation tests:

```powershell
python -m unittest discover -s tests
```

## Demo Highlights

When you present the project, these are the strongest SRM-focused features to show:

- event-aware SRM scenarios such as `Sports Event`, `Exam Rush`, and `Class Changeover`
- planner / critic / executor decisions in the dashboard agent loop
- adaptive learning with persisted reward and Q-table updates
- proactive notifications through the mock delivery feed
- measurable KPI improvements such as search time, utilisation, and allocation success across SRM blocks

## Security Note

- Never commit your real `.env` file or API keys.
- Rotate any API key that has already been exposed.
- Use placeholder keys only in `.env.example`.
