import streamlit as st
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ LOAD ENV
from dotenv import load_dotenv
load_dotenv()

# SYSTEM IMPORTS
from environment.parking_environment import ParkingEnvironment
from agents.demand_agent import DemandAgent
from agents.monitoring_agent import MonitoringAgent
from agents.policy_agent import PolicyAgent
from agents.bayesian_agent import BayesianAgent
from agents.reward_agent import RewardAgent
from agent_memory import AgentMemory

# ✅ GEMINI
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------ API KEY FIX ------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Check .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

def ask_llm(prompt):
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"LLM Error: {e}"

# ------------------ STREAMLIT ------------------
st.set_page_config(layout="wide")
st.title("🚗 Intelligent Agentic Parking System")

# ------------------ SESSION STATE ------------------
if "env" not in st.session_state:
    st.session_state.env = ParkingEnvironment()

if "memory" not in st.session_state:
    st.session_state.memory = AgentMemory()

if "run" not in st.session_state:
    st.session_state.run = False

if "reasoning" not in st.session_state:
    st.session_state.reasoning = ""

env = st.session_state.env
memory = st.session_state.memory

# ------------------ AGENTS ------------------
demand_agent = DemandAgent()
monitor = MonitoringAgent()
policy = PolicyAgent()
bayesian = BayesianAgent()
reward = RewardAgent()

# ------------------ TOGGLE ------------------
auto = st.toggle("Autonomous Mode")

st.session_state.run = auto

# ------------------ MAIN LOOP ------------------
if st.session_state.run:

    state = env.get_state()

    monitored = monitor.observe(state)
    demand = demand_agent.predict()
    insight = bayesian.infer(monitored)

    action = policy.decide(monitored, demand, insight)

    if action:
        env.apply_action(action)

    new_state = env.step()

    reward.evaluate(state, new_state)
    memory.add(new_state)

    # ✅ GEMINI REASONING
    prompt = f"""
    Parking system state:
    {new_state}

    Tell:
    - Most crowded area
    - Best parking area
    - Short reasoning
    """

    st.session_state.reasoning = ask_llm(prompt)

    time.sleep(1)
    st.rerun()

# ------------------ DISPLAY ------------------
state = env.get_state()

st.subheader("📊 Live Status")
st.table(state)

# ------------------ ALERTS ------------------
st.subheader("🚨 Alerts")

for z in state:
    if state[z]["free_slots"] <= 5:
        st.error(f"{z} FULL 🚨")
    elif state[z]["free_slots"] <= 10:
        st.warning(f"{z} almost full")

# ------------------ BAR CHART ------------------
st.subheader("📊 Free Slots")
st.bar_chart([state[z]["free_slots"] for z in state])

# ------------------ TREND ------------------
st.subheader("📈 Trend")

trend = env.get_trend()

if trend:
    for z in trend[0]:
        st.line_chart([t[z] for t in trend])

# ------------------ METRICS ------------------
st.subheader("📉 Performance Metrics")
st.write(memory.get_metrics())

# ------------------ AI REASONING ------------------
st.subheader("🧠 Autonomous AI Thinking")

if st.session_state.reasoning:
    st.info(st.session_state.reasoning)

# ------------------ CHAT ------------------
st.subheader("💬 AI Chat (Gemini)")

user_query = st.text_input("Ask anything about parking...")

if user_query:
    prompt = f"""
    Parking data:
    {state}

    User question: {user_query}

    Answer clearly and intelligently.
    """

    response = ask_llm(prompt)
    st.success(response)

# ------------------ DEBUG ------------------
st.write("DEBUG:", state)