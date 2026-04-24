import streamlit as st
import plotly.express as px
import pandas as pd

def build_zone_chart(state_frame):
    if state_frame.empty:
        return None
    fig = px.bar(
        state_frame,
        x="Zone",
        y=["Occupied", "Free"],
        barmode="group",
        title="SRM Block Occupancy vs Free Capacity",
        labels={"Zone": "SRM Block", "value": "Slots", "variable": "Metric"},
        color_discrete_sequence=["#8dc8ff", "#1f74cc"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def build_utilisation_chart(state_frame):
    if state_frame.empty:
        return None
    frame = state_frame.copy()
    fig = px.bar(
        frame.sort_values("Utilisation %", ascending=False),
        x="Utilisation %",
        y="Zone",
        orientation="h",
        title="SRM Block Congestion Priority",
        color="Utilisation %",
        labels={"Zone": "SRM Block"},
        color_continuous_scale=["#4bd38a", "#d8c86e", "#ff746c"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig

def build_flow_chart(recent_states):
    if not recent_states:
        return None
    rows = []
    redirect_steps = []
    for item in recent_states:
        totals = item.get("transition", {}).get("totals", {})
        step = item.get("step", 0)
        rows.append(
            {
                "step": step,
                "Entries": totals.get("entries", 0),
                "Exits": totals.get("exits", 0),
                "Redirected": item.get("transition", {}).get("transferred", 0),
            }
        )
        action = item.get("action", {}) or item.get("transition", {}).get("applied_action", {})
        if action.get("action") == "redirect":
            redirect_steps.append(step)
            
    frame = pd.DataFrame(rows).melt(id_vars="step", var_name="metric", value_name="vehicles")
    fig = px.area(
        frame,
        x="step",
        y="vehicles",
        color="metric",
        title="Vehicle Movement Over Time",
        color_discrete_sequence=["#8dc8ff", "#388ae5", "#ff9d91"],
    )
    
    for step_num in redirect_steps:
        fig.add_vline(x=step_num, line_width=2, line_dash="dash", line_color="rgba(75, 211, 138, 0.45)", annotation_text="Action", annotation_position="top left", annotation_font_color="rgba(75, 211, 138, 0.8)")
        
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def build_kpi_chart(recent_states):
    if not recent_states:
        return None
    rows = []
    redirect_steps = []
    for index, item in enumerate(recent_states):
        if not isinstance(item, dict):
            continue
        kpis = item.get("kpis", {}) if isinstance(item.get("kpis", {}), dict) else {}
        step = item.get("step", index)
        rows.append(
            {
                "step": step,
                "Search Time": float(kpis.get("estimated_search_time_min", 0) or 0),
                "SRM Block Hotspots": float(kpis.get("congestion_hotspots", 0) or 0),
            }
        )
        transition = item.get("transition", {}) if isinstance(item.get("transition", {}), dict) else {}
        action = item.get("action", {}) or transition.get("applied_action", {})
        if action.get("action") == "redirect":
            redirect_steps.append(step)
    if not rows:
        return None

    frame = pd.DataFrame(rows).melt(id_vars="step", var_name="metric", value_name="value")
    if frame.empty:
        return None
    fig = px.line(
        frame,
        x="step",
        y="value",
        color="metric",
        markers=True,
        title="Search Time and Congestion Trend",
        color_discrete_sequence=["#ff9d91", "#8dc8ff"],
    )
    
    for step_num in redirect_steps:
        fig.add_vline(x=step_num, line_width=2, line_dash="dash", line_color="rgba(75, 211, 138, 0.45)", annotation_text="Agent Action", annotation_position="top left", annotation_font_color="rgba(75, 211, 138, 0.8)")
        
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def build_benchmark_chart(frame):
    if frame.empty:
        return None
    melted = frame.melt(
        id_vars="Scenario",
        value_vars=["Agentic Search Time", "Baseline Search Time"],
        var_name="Mode",
        value_name="Minutes",
    )
    fig = px.bar(
        melted,
        x="Scenario",
        y="Minutes",
        color="Mode",
        barmode="group",
        title="Agentic vs Baseline Search Time",
        color_discrete_sequence=["#4bd38a", "#ff746c"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def build_latest_baseline_chart(baseline_comparison):
    if not baseline_comparison:
        return None
    baseline = baseline_comparison.get("baseline_kpis", {})
    agent = baseline_comparison.get("agent_kpis", {})
    if not baseline or not agent:
        return None

    rows = []
    metric_map = [
        ("Search Time", "estimated_search_time_min"),
        ("SRM Block Hotspots", "congestion_hotspots"),
    ]
    for label, key in metric_map:
        rows.append({"Metric": label, "Mode": "Agent", "Value": agent.get(key, 0)})
        rows.append({"Metric": label, "Mode": "No-Redirect Baseline", "Value": baseline.get(key, 0)})

    fig = px.bar(
        pd.DataFrame(rows),
        x="Metric",
        y="Value",
        color="Mode",
        barmode="group",
        title="Current Step: Agent vs No-Redirect Baseline",
        color_discrete_sequence=["#4bd38a", "#ff746c"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def build_performance_trend_chart(recent_states):
    """Visualizes the adaptive gain (Search Time Reduction) over the last 20 steps."""
    if len(recent_states) < 5:
        return None
        
    rows = []
    for item in recent_states:
        transition = item.get("transition", {})
        baseline = transition.get("baseline_comparison", {})
        delta = baseline.get("search_time_delta_min", 0.0)
        
        # Calculate a moving average or just raw delta
        rows.append({
            "Step": item.get("step", 0),
            "Search Time Gain (Min)": delta,
            "Mode": item.get("planner_output", {}).get("decision_mode", "deterministic")
        })
        
    df = pd.DataFrame(rows)
    if df.empty:
        return None
        
    fig = px.line(
        df,
        x="Step",
        y="Search Time Gain (Min)",
        title="Learning Adaptation Trend",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=["#4bd38a"]
    )
    
    # Add a zero baseline for clarity
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Simulation Step",
        yaxis_title="Minutes Saved vs Baseline",
    )
    return fig
    return fig
