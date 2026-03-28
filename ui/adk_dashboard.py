import os
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from agent_controller import AgentController
from agent_memory import AgentMemory
from environment.parking_environment import ParkingEnvironment
from llm_reasoning import get_llm, get_local_chat_response, summarize_state


st.set_page_config(layout="wide", page_title="Intelligent Parking System")
st.title("Intelligent Agentic Parking System")


def _format_state_table(state):
    frame = pd.DataFrame(state).T
    return frame[["total_slots", "occupied", "free_slots", "entry", "exit"]]


if "env" not in st.session_state:
    st.session_state.env = ParkingEnvironment()
if "memory" not in st.session_state:
    st.session_state.memory = AgentMemory()
if "controller" not in st.session_state:
    st.session_state.controller = AgentController(environment=st.session_state.env)
if "run" not in st.session_state:
    st.session_state.run = False
if "last_run" not in st.session_state:
    st.session_state.last_run = 0.0
if "agent_data" not in st.session_state:
    st.session_state.agent_data = {}
if "reasoning" not in st.session_state:
    st.session_state.reasoning = ""
if "latest_reward" not in st.session_state:
    st.session_state.latest_reward = 0.0
if "chat_response" not in st.session_state:
    st.session_state.chat_response = ""

env = st.session_state.env
memory = st.session_state.memory
controller = st.session_state.controller
llm = get_llm()

st.subheader("Controls")
auto = st.toggle("Autonomous Mode", value=st.session_state.run)
st.session_state.run = auto
speed = st.slider("Simulation Speed (seconds)", 0.2, 3.0, 1.0)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start"):
        st.session_state.run = True
with col2:
    if st.button("Pause"):
        st.session_state.run = False
with col3:
    if st.button("Reset Simulation"):
        st.session_state.env = ParkingEnvironment()
        st.session_state.controller = AgentController(environment=st.session_state.env)
        st.session_state.memory = st.session_state.controller.memory
        st.session_state.agent_data = {}
        st.session_state.reasoning = ""
        st.session_state.latest_reward = 0.0
        st.session_state.chat_response = ""
        st.rerun()

current_time = time.time()

if st.session_state.run and (current_time - st.session_state.last_run > speed):
    st.session_state.last_run = current_time
    result = controller.step()
    new_state = result["state"]

    st.session_state.latest_reward = result["environment_reward"]
    st.session_state.agent_data = {
        "mode": result["mode"],
        "demand": result["demand"],
        "monitored": new_state,
        "insight": result["insight"],
        "llm_action": result["llm_action"],
        "final_action": result["action"],
        "reward_score": result["reward_score"],
        "summary": result["summary"],
    }
    st.session_state.reasoning = result["reasoning"]
    st.session_state.memory = controller.memory
    st.rerun()

env = st.session_state.controller.environment
memory = st.session_state.controller.memory
state = env.get_state()
state_frame = _format_state_table(state)
summary = summarize_state(state)
most_crowded = summary["most_crowded"]
best_zone = summary["best_zone"]
total_free = sum(state[zone]["free_slots"] for zone in state)
total_capacity = sum(state[zone]["total_slots"] for zone in state)
congestion = 100 - (total_free / total_capacity * 100) if total_capacity else 0

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Congestion Level (%)", round(congestion, 2))
metric2.metric("Latest Environment Reward", st.session_state.latest_reward)
metric3.metric("Best Zone", best_zone)

st.subheader("Live Status")
st.dataframe(state_frame, use_container_width=True)

st.subheader("Quick Insight")
quick1, quick2 = st.columns(2)
quick1.error(f"Most Crowded: {most_crowded}")
quick2.success(f"Best Parking: {best_zone}")

st.subheader("Zone Status")
status_cols = st.columns(len(state))
for index, zone in enumerate(state):
    free = state[zone]["free_slots"]
    if free <= 5:
        status_cols[index].error(f"{zone}\n{free} free")
    elif free <= 15:
        status_cols[index].warning(f"{zone}\n{free} free")
    else:
        status_cols[index].success(f"{zone}\n{free} free")

st.subheader("Alerts")
alert_count = 0
for zone, data in state.items():
    if data["free_slots"] <= 3:
        alert_count += 1
        st.error(f"{zone} is nearly full. Entry: {data['entry']}, Exit: {data['exit']}")
    elif data["free_slots"] <= 10:
        alert_count += 1
        st.warning(f"{zone} is approaching congestion.")
if alert_count == 0:
    st.success("No critical congestion alerts right now.")

st.subheader("Free Slots")
st.bar_chart(state_frame["free_slots"])

st.subheader("Trend")
trend = env.get_trend()
if trend:
    trend_frame = pd.DataFrame(
        [{zone: snapshot[zone]["free_slots"] for zone in snapshot} for snapshot in trend]
    )
    st.line_chart(trend_frame)

st.subheader("Performance Metrics")
st.json(memory.get_metrics())

st.subheader("Agent Communication")
agent_data = st.session_state.get("agent_data", {})
if agent_data:
    st.write("Mode:", agent_data.get("mode"))
    st.write("Demand Agent:", {
        zone: f"{value}/100"
        for zone, value in agent_data.get("demand", {}).items()
    })
    st.write("Monitoring Agent:", agent_data.get("monitored"))
    st.write("Bayesian Agent:", agent_data.get("insight"))
    st.write("LLM Decision:", agent_data.get("llm_action"))
    st.write("Final Action:", agent_data.get("final_action"))
    st.write("Reward Agent Score:", agent_data.get("reward_score"))
else:
    st.info("Start autonomous mode to see agent communication.")

st.subheader("Autonomous AI Thinking")
st.info(st.session_state.reasoning or "Run the simulation to generate reasoning.")

st.subheader("AI Chat")

with st.form("chat_form", clear_on_submit=False):
    query = st.text_input("Ask anything about parking...")
    submitted = st.form_submit_button("Ask")

if submitted and query.strip():
    if llm is None:
        st.session_state.chat_response = get_local_chat_response(state, query.strip())
    else:
        try:
            response = llm.invoke(
                f"Parking state: {state}\nUser question: {query.strip()}\n"
                "Answer in 2-3 short sentences and use the state values."
            ).content
            st.session_state.chat_response = response
        except Exception:
            st.warning("LLM request failed. Falling back to local reasoning.")
            st.session_state.chat_response = get_local_chat_response(state, query.strip())

if st.session_state.chat_response:
    st.success(st.session_state.chat_response)
