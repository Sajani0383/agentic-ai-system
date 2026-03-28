# Smart Parking Agentic AI System

An agentic AI prototype for smart parking management with:

- multi-zone parking simulation
- autonomous agent controller
- demand estimation and congestion-aware routing
- Streamlit dashboard
- FastAPI service
- optional Gemini-backed reasoning

## Project Structure

- `simulation.py`: CLI simulation entry point
- `agent_controller.py`: main orchestration layer for agents and environment
- `environment/parking_environment.py`: parking environment and reward logic
- `agents/`: monitoring, demand, Bayesian, policy, and reward agents
- `ui/adk_dashboard.py`: Streamlit dashboard
- `adk/agent_api.py`: FastAPI app
- `ml/`: demand prediction model loading and training
- `start_project.ps1`: start API and UI
- `stop_project.ps1`: stop API and UI

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
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

## Optional LLM Mode

The project is safe by default in offline mode.

To enable Gemini calls, add this to `.env`:

```env
ENABLE_LLM=true
```

If `ENABLE_LLM` is not set, the system uses local fallback reasoning so the app still works reliably.

## Security Note

- Never commit your real `.env` file or API keys.
- Rotate any API key that has already been exposed.
- Keep `ENABLE_LLM=false` unless you intentionally want live external model calls.
