#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8501}"

mkdir -p logs

echo "Starting SRM Smart Parking API on http://${API_HOST}:${API_PORT}"
"$PYTHON_BIN" -m uvicorn adk.agent_api:app --host "$API_HOST" --port "$API_PORT" > logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid

echo "Starting Streamlit dashboard on http://${UI_HOST}:${UI_PORT}"
"$PYTHON_BIN" -m streamlit run ui/adk_dashboard.py --server.headless true --server.address "$UI_HOST" --server.port "$UI_PORT" > logs/dashboard.log 2>&1 &
UI_PID=$!
echo "$UI_PID" > logs/dashboard.pid

echo "API PID: $API_PID"
echo "Dashboard PID: $UI_PID"
echo "Stop with: kill \$(cat logs/api.pid) \$(cat logs/dashboard.pid)"
