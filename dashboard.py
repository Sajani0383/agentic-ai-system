import random
import time

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Smart Parking AI", layout="wide")
st.title("Smart Parking Agentic AI Dashboard")

dataset = pd.read_csv("dataset/parking_dataset.csv")
dataset["free_slots"] = dataset["total_slots"] - dataset["occupied_slots"]
zones = dataset["zone"].dropna().unique().tolist()

st.sidebar.header("Simulation Controls")
vehicle_count = st.sidebar.slider("Vehicles", 10, 100, 40)
speed = st.sidebar.slider("Simulation Speed", 0.1, 1.0, 0.4)
start = st.sidebar.button("Start Simulation")

col1, col2, col3 = st.columns(3)
col1.metric("Parking Zones", len(zones))
col2.metric("Vehicles", vehicle_count)
col3.metric("AI Agent", "Active")

st.divider()
st.subheader("Parking Availability")
zone_data = dataset.groupby("zone", as_index=False)["free_slots"].mean()
fig = px.bar(
    zone_data,
    x="zone",
    y="free_slots",
    color="zone",
    title="Average Free Parking Slots",
)
st.plotly_chart(fig, use_container_width=True)

if start:
    st.divider()
    st.subheader("Live AI Decision")

    decision_box = st.empty()
    reward_box = st.empty()
    decision_history = []
    rewards = []

    for step in range(20):
        zone_snapshot = random.choice(zones)
        reason = random.choice(
            [
                "Low congestion pressure",
                "High slot availability",
                "Balanced traffic flow",
            ]
        )
        confidence = round(random.uniform(0.75, 0.95), 2)
        reward = random.randint(-1, 3)

        decision_box.markdown(
            f"""
### Step {step}

**Chosen Zone:** {zone_snapshot}  
**Reason:** {reason}  
**Confidence:** {confidence}
"""
        )
        reward_box.metric("Latest Reward", reward)

        rewards.append(reward)
        decision_history.append(
            {
                "Step": step,
                "Zone": zone_snapshot,
                "Reason": reason,
                "Confidence": confidence,
                "Reward": reward,
            }
        )

        time.sleep(speed)

    st.divider()
    st.subheader("Reinforcement Learning Reward")
    reward_df = pd.DataFrame({"Step": range(len(rewards)), "Reward": rewards})
    fig2 = px.line(reward_df, x="Step", y="Reward", title="Reward Trend")
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Agent Decision Summary")
    decision_df = pd.DataFrame(decision_history)
    st.dataframe(decision_df, use_container_width=True)

st.divider()
st.subheader("Vehicle Entry vs Exit")
entry_exit = dataset[["entry_count", "exit_count"]]
fig3 = px.line(entry_exit, title="Entry vs Exit Flow")
st.plotly_chart(fig3, use_container_width=True)
