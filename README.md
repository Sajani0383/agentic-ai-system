# Smart Parking Agentic AI System

A simulation-first agentic AI parking project designed for campus use cases, with:

- event-aware campus parking simulation
- planner / critic / executor agent loop
- adaptive reward learning plus Q-learning policy updates
- demand estimation and congestion-aware routing
- Streamlit command-center dashboard
- FastAPI runtime and mock notification API
- optional Gemini-backed reasoning when API keys are available

## Project Structure

- `simulation.py`: CLI simulation entry point
- `agent_controller.py`: main orchestration layer for agents and environment
- `environment/parking_environment.py`: parking environment and reward logic
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

## Simulation-First Mode

This project is complete as a simulation even without real hardware, live feeds, or external APIs.

- demand prediction falls back safely if the saved model cannot be loaded
- planner and critic continue through tool-driven local reasoning if no LLM key is available
- notifications are delivered through a mock API feed so the dashboard still demonstrates proactive action

## Optional LLM Mode

If you add a valid Gemini key in `.env`, the planner and critic will prefer live model reasoning:

```env
ENABLE_LLM=true
GOOGLE_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Without a valid key, the project still runs as a fully functional simulation.

Check Gemini directly:

```bash
python3 -c "import llm_reasoning; llm=llm_reasoning.get_llm(); print('llm', bool(llm)); print(llm.invoke('Reply with exactly: ok').content if llm else 'no llm')"
```

If this prints `ok`, the Gemini key and model are working. If it fails with `nodename nor servname provided`, the code is configured but the current runtime cannot reach the Gemini API because DNS/network access is blocked.

## Tests

Run the simulation validation tests:

```powershell
python -m unittest discover -s tests
```

## Demo Highlights

When you present the project, these are the strongest features to show:

- event-aware campus scenarios such as `Sports Event`, `Exam Rush`, and `Class Changeover`
- planner / critic / executor decisions in the dashboard agent loop
- adaptive learning with persisted reward and Q-table updates
- proactive notifications through the mock delivery feed
- measurable KPI improvements such as search time, utilisation, and allocation success

## Security Note

- Never commit your real `.env` file or API keys.
- Rotate any API key that has already been exposed.
- Use placeholder keys only in `.env.example`.
