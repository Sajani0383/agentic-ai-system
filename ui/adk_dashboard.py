import time

import pandas as pd
import plotly.express as px
import streamlit as st

from services.parking_runtime import runtime_service


PALETTE = {
    "bg": "#081018",
    "panel": "#101b2b",
    "border": "#20334b",
    "text": "#eef4ff",
    "muted": "#95a6bd",
    "blue": "#4da3ff",
    "green": "#4bd38a",
    "gold": "#d8c86e",
    "coral": "#ff746c",
    "cyan": "#6edff6",
    "violet": "#9f8cff",
}

SCENARIOS = [
    "Auto Schedule",
    "Normal Day",
    "Class Changeover",
    "Exam Rush",
    "Sports Event",
    "Fest Night",
    "Emergency Spillover",
]


def _ensure_session_state():
    if "run" not in st.session_state:
        st.session_state.run = False
    if "last_run" not in st.session_state:
        st.session_state.last_run = 0.0
    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""


def _schedule_reload(seconds):
    # Keep autonomous mode on the server side so the simulation advances
    # even when the browser ignores or delays injected JS reloads.
    time.sleep(max(0.2, seconds))
    st.rerun()


def _inject_styles():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(77,163,255,0.12), transparent 28%),
                linear-gradient(180deg, #09111b 0%, {PALETTE["bg"]} 100%);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1c202b 0%, #222632 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        .hero-title {{
            font-size: 3.35rem;
            line-height: 1.02;
            font-weight: 800;
            color: {PALETTE["text"]};
            letter-spacing: -0.04em;
            margin-bottom: 0.45rem;
        }}
        .hero-sub {{
            color: {PALETTE["muted"]};
            font-size: 1.03rem;
            max-width: 56rem;
        }}
        .event-banner {{
            margin: 1rem 0 0.9rem 0;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, rgba(77,163,255,0.16), rgba(216,200,110,0.10));
            box-shadow: 0 18px 50px rgba(0,0,0,0.16);
        }}
        .event-title {{
            color: {PALETTE["text"]};
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .event-copy {{
            color: {PALETTE["muted"]};
            font-size: 0.96rem;
        }}
        .signal-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.7rem 0 1.1rem;
        }}
        .signal-card {{
            background: linear-gradient(180deg, rgba(18,31,47,0.94), rgba(11,22,34,0.94));
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 16px 50px rgba(0,0,0,0.22);
        }}
        .signal-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE["muted"]};
            margin-bottom: 0.35rem;
        }}
        .signal-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {PALETTE["text"]};
            line-height: 1.15;
        }}
        .signal-note {{
            margin-top: 0.42rem;
            color: {PALETTE["muted"]};
            font-size: 0.9rem;
        }}
        .story-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.95rem;
            margin: 0.9rem 0 1.1rem;
        }}
        .story-card {{
            background: linear-gradient(180deg, rgba(16,27,43,0.96), rgba(10,18,29,0.96));
            border: 1px solid {PALETTE["border"]};
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 16px 44px rgba(0,0,0,0.24);
        }}
        .story-title {{
            color: {PALETTE["muted"]};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            margin-bottom: 0.45rem;
        }}
        .story-value {{
            color: {PALETTE["text"]};
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }}
        .story-copy {{
            color: {PALETTE["muted"]};
            font-size: 0.95rem;
            line-height: 1.55;
        }}
        .zone-grid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.6rem 0 1rem;
        }}
        .zone-card {{
            background: linear-gradient(180deg, rgba(14,24,37,0.96), rgba(8,16,26,0.96));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 20px;
            padding: 0.95rem;
        }}
        .zone-name {{
            color: {PALETTE["text"]};
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.28rem;
        }}
        .zone-meta {{
            color: {PALETTE["muted"]};
            font-size: 0.84rem;
            margin-bottom: 0.7rem;
        }}
        .util-track {{
            width: 100%;
            height: 9px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.55rem 0 0.35rem;
        }}
        .util-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        .zone-stat-row {{
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            color: {PALETTE["muted"]};
            font-size: 0.83rem;
            margin-top: 0.35rem;
        }}
        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }}
        .chip-blue {{
            background: rgba(77,163,255,0.12);
            color: #8ec8ff;
            border-color: rgba(77,163,255,0.25);
        }}
        .chip-green {{
            background: rgba(75,211,138,0.12);
            color: #81e6aa;
            border-color: rgba(75,211,138,0.25);
        }}
        .chip-gold {{
            background: rgba(216,200,110,0.12);
            color: #ebdf91;
            border-color: rgba(216,200,110,0.25);
        }}
        .chip-coral {{
            background: rgba(255,116,108,0.12);
            color: #ff9e98;
            border-color: rgba(255,116,108,0.25);
        }}
        .feature-callout {{
            background: linear-gradient(135deg, rgba(110,223,246,0.10), rgba(159,140,255,0.08));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            margin: 0.5rem 0 1rem;
        }}
        .section-kicker {{
            color: {PALETTE["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            margin-bottom: 0.28rem;
        }}
        .section-copy {{
            color: {PALETTE["muted"]};
            margin-top: -0.2rem;
            margin-bottom: 0.75rem;
            font-size: 0.93rem;
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, rgba(15,27,41,0.95), rgba(10,18,29,0.95));
            border: 1px solid {PALETTE["border"]};
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
        }}
        div[data-testid="stMetricLabel"] {{ color: {PALETTE["muted"]}; }}
        div[data-testid="stMetricValue"] {{ color: {PALETTE["text"]}; }}
        div[data-testid="stDataFrame"], div[data-testid="stJson"] {{
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            overflow: hidden;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.7rem; }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.45rem 0.9rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(77,163,255,0.12);
            border-color: rgba(77,163,255,0.35);
        }}
        @media (max-width: 1000px) {{
            .hero-title {{ font-size: 2.4rem; }}
            .signal-grid {{ grid-template-columns: 1fr; }}
            .story-grid {{ grid-template-columns: 1fr; }}
            .zone-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_state_table(state):
    frame = pd.DataFrame(state).T.reset_index(names="Zone")
    frame = frame.rename(
        columns={
            "total_slots": "Capacity",
            "occupied": "Occupied",
            "free_slots": "Free",
            "entry": "Entries",
            "exit": "Exits",
        }
    )
    frame["Utilisation %"] = ((frame["Occupied"] / frame["Capacity"]) * 100).round(1)
    frame["Recommendation"] = frame["Free"].apply(
        lambda free: "Preferred" if free >= 20 else ("Overflow" if free >= 10 else "Avoid")
    )
    return frame[["Zone", "Capacity", "Occupied", "Free", "Entries", "Exits", "Utilisation %", "Recommendation"]]


def _utilisation_color(utilisation):
    if utilisation >= 90:
        return PALETTE["coral"]
    if utilisation >= 80:
        return PALETTE["gold"]
    if utilisation >= 65:
        return PALETTE["blue"]
    return PALETTE["green"]


def _recommendation_tone(label):
    if label == "Avoid":
        return "coral"
    if label == "Overflow":
        return "gold"
    return "green"


def _signal_cards(event_context, latest_result, kpis, goal):
    action = latest_result.get("action", {})
    critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
    action_reason = action.get("reason")
    if not action_reason:
        action_reason = critic_notes[0] if critic_notes else "The agent is holding position because current pressure does not justify a transfer."
    return [
        {
            "label": "Current Event",
            "value": event_context.get("name", "Normal Day"),
            "note": f"Severity: {event_context.get('severity', 'low').title()} | Focus: {event_context.get('focus_zone', '-')}",
        },
        {
            "label": "Allocation Strategy",
            "value": latest_result.get("strategy", event_context.get("allocation_strategy", "Balanced utilisation")),
            "note": f"Recommended overflow zone: {event_context.get('recommended_zone', '-')}",
        },
        {
            "label": "Live Decision",
            "value": f"{action.get('action', 'none').upper()} | {latest_result.get('mode', 'idle').replace('_', ' ')}",
            "note": action_reason,
        },
        {
            "label": "Measured Outcome",
            "value": f"{kpis.get('estimated_search_time_min', 0)} min search time",
            "note": f"Allocation success: {kpis.get('allocation_success_pct', 0)}% | Utilisation: {kpis.get('space_utilisation_pct', 0)}%",
        },
    ]


def _transition_frame(latest_transition):
    zone_rows = latest_transition.get("zones", [])
    if not zone_rows:
        return pd.DataFrame(columns=["Zone", "Before", "After", "Entries", "Exits", "Net Change"])
    frame = pd.DataFrame(zone_rows).rename(
        columns={
            "zone": "Zone",
            "occupied_before": "Before",
            "occupied_after": "After",
            "entry": "Entries",
            "exit": "Exits",
            "occupancy_change": "Net Change",
        }
    )
    return frame[["Zone", "Before", "After", "Entries", "Exits", "Net Change"]]


def _agent_frame(latest_result):
    rows = []
    for item in latest_result.get("agent_interactions", []):
        payload = item.get("payload", {})
        if isinstance(payload, dict):
            preview = ", ".join(f"{key}: {value}" for key, value in list(payload.items())[:3])
        else:
            preview = str(payload)
        rows.append(
            {
                "Agent": item.get("agent"),
                "Action Taken": item.get("message"),
                "Key Output": preview[:130],
            }
        )
    return pd.DataFrame(rows)


def _cycle_frame(recent_cycles):
    return pd.DataFrame(
        [
            {
                "Step": cycle.get("step"),
                "Event": cycle.get("event_context", {}).get("name", ""),
                "Goal": cycle.get("goal", {}).get("objective", ""),
                "Planner": cycle.get("planner_output", {}).get("proposed_action", {}).get("action", "none").upper(),
                "Final": cycle.get("execution_output", {}).get("final_action", {}).get("action", "none").upper(),
                "Reward": cycle.get("reward", {}).get("environment_reward", 0),
            }
            for cycle in recent_cycles
        ]
    )


def _trace_frame(trace):
    return pd.DataFrame(
        [
            {
                "Step": str(item.get("step", "-")),
                "Mode": str(item.get("mode", item.get("type", "-"))),
                "Action": str(item.get("action", item.get("query", "-")))[:60],
                "Goal / Output": str(item.get("goal", item.get("answer", "-")))[:90],
            }
            for item in trace
        ]
    )


def _build_zone_chart(state_frame):
    fig = px.bar(
        state_frame,
        x="Zone",
        y=["Occupied", "Free"],
        barmode="group",
        title="Current Occupancy vs Free Capacity",
        color_discrete_sequence=["#8dc8ff", "#1f74cc"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _build_utilisation_chart(state_frame):
    frame = state_frame.copy()
    fig = px.bar(
        frame.sort_values("Utilisation %", ascending=False),
        x="Utilisation %",
        y="Zone",
        orientation="h",
        title="Congestion Priority by Zone",
        color="Utilisation %",
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


def _build_flow_chart(recent_states):
    if not recent_states:
        return None
    rows = []
    for item in recent_states:
        totals = item.get("transition", {}).get("totals", {})
        rows.append(
            {
                "step": item.get("step", 0),
                "Entries": totals.get("entries", 0),
                "Exits": totals.get("exits", 0),
                "Redirected": item.get("transition", {}).get("transferred", 0),
            }
        )
    frame = pd.DataFrame(rows).melt(id_vars="step", var_name="metric", value_name="vehicles")
    fig = px.area(
        frame,
        x="step",
        y="vehicles",
        color="metric",
        title="Vehicle Movement Over Time",
        color_discrete_sequence=["#8dc8ff", "#388ae5", "#ff9d91"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _build_kpi_chart(recent_states):
    if not recent_states:
        return None
    rows = []
    for item in recent_states:
        kpis = item.get("kpis", {})
        rows.append(
            {
                "step": item.get("step", 0),
                "search_time": kpis.get("estimated_search_time_min", 0),
                "utilisation": kpis.get("space_utilisation_pct", 0),
                "allocation_success": kpis.get("allocation_success_pct", 0),
            }
        )
    frame = pd.DataFrame(rows).melt(id_vars="step", var_name="metric", value_name="value")
    fig = px.line(
        frame,
        x="step",
        y="value",
        color="metric",
        markers=True,
        title="Industry KPI Trend",
        color_discrete_sequence=["#ff9d91", "#8dc8ff", "#4bd38a"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _build_benchmark_frame(benchmark):
    rows = []
    for item in benchmark.get("scenarios", []):
        rows.append(
            {
                "Scenario": item.get("scenario"),
                "Agentic Search Time": item.get("agentic", {}).get("avg_search_time_min", 0),
                "Baseline Search Time": item.get("baseline", {}).get("avg_search_time_min", 0),
                "Search Time Gain": item.get("delta_search_time", 0),
                "Agentic Resilience": item.get("agentic", {}).get("avg_resilience_score", 0),
                "Baseline Resilience": item.get("baseline", {}).get("avg_resilience_score", 0),
                "Resilience Gain": item.get("delta_resilience", 0),
                "Hotspot Reduction": item.get("delta_hotspots", 0),
            }
        )
    return pd.DataFrame(rows)


def _build_benchmark_chart(frame):
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


def _render_notifications(notifications):
    if not notifications:
        st.info("No active campus alerts right now.")
        return
    for notification in notifications:
        text = f"**{notification.get('title', 'Update')}**  \n{notification.get('message', '')}"
        level = notification.get("level")
        if level == "error":
            st.error(text)
        elif level == "warning":
            st.warning(text)
        elif level == "success":
            st.success(text)
        else:
            st.info(text)


def _render_html_block(html):
    st.markdown(html, unsafe_allow_html=True)


def _render_status_callout(title, body, level="info"):
    if level == "success":
        st.success(f"**{title}**\n\n{body}")
    elif level == "warning":
        st.warning(f"**{title}**\n\n{body}")
    elif level == "error":
        st.error(f"**{title}**\n\n{body}")
    else:
        st.info(f"**{title}**\n\n{body}")


def _render_assistant_briefing(briefing):
    if not briefing:
        return
    st.markdown("**Live AI Copilot**")
    st.info(f"**{briefing.get('headline', 'Operations summary')}**\n\n{briefing.get('narrative', '')}")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Prediction**")
        st.write(briefing.get("prediction", "No prediction available yet."))
    with cols[1]:
        st.markdown("**Decision Commentary**")
        st.write(briefing.get("decision_commentary", "No decision commentary available yet."))
    suggestions = briefing.get("suggestions", [])
    if suggestions:
        st.markdown("**Suggested Next Moves**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def _render_story_cards(goal, latest_result, event_context, kpis):
    action = latest_result.get("action", {})
    story_cards = [
        {
            "title": "Why The Agent Acted",
            "value": event_context.get("focus_zone", "Campus"),
            "copy": action.get("reason", "The agent is waiting for enough demand pressure to justify reallocation."),
        },
        {
            "title": "Active Goal",
            "value": f"{goal.get('target_congested_zones', '-') } zone target",
            "copy": goal.get("objective", "No active goal yet."),
        },
        {
            "title": "Student Outcome",
            "value": f"{kpis.get('estimated_search_time_min', 0)} min",
            "copy": "Estimated parking search time after the current allocation decision.",
        },
    ]
    _render_html_block(
        "<div class='story-grid'>"
        + "".join(
            f"<div class='story-card'><div class='story-title'>{card['title']}</div>"
            f"<div class='story-value'>{card['value']}</div>"
            f"<div class='story-copy'>{card['copy']}</div></div>"
            for card in story_cards
        )
        + "</div>"
    )


def _render_zone_cards(state_frame):
    zone_columns = st.columns(len(state_frame))
    for column, (_, row) in zip(zone_columns, state_frame.iterrows()):
        utilisation = float(row["Utilisation %"])
        tone = _recommendation_tone(row["Recommendation"])
        with column:
            st.markdown(f"**{row['Zone']}**")
            st.caption(f"Capacity {int(row['Capacity'])} · Occupied {int(row['Occupied'])}")
            if tone == "coral":
                st.error(row["Recommendation"])
            elif tone == "gold":
                st.warning(row["Recommendation"])
            else:
                st.success(row["Recommendation"])
            st.progress(min(max(utilisation / 100, 0), 1.0), text=f"{utilisation:.1f}% utilised")
            st.markdown(f"`Free slots` {int(row['Free'])}")
            st.markdown(f"`Entries / Exits` {int(row['Entries'])} / {int(row['Exits'])}")


def _build_llm_summary(llm_status):
    if llm_status.get("available"):
        message = llm_status.get("message", "Gemini SDK is configured.")
        if llm_status.get("last_error"):
            return (
                "Gemini Configured, Fallback Active",
                f"Model: {llm_status.get('model', 'gemini')}. {message} Last error: {llm_status.get('last_error', 'n/a')}. The dashboard is using local fallback reasoning until the next Gemini call succeeds.",
                "warning",
            )
        return (
            "Gemini Ready",
            f"Model: {llm_status.get('model', 'gemini')}. Env: {llm_status.get('env_path', '.env')}. Planner, critic, and chat are set to use Gemini first, with local fallback only if the live call fails.",
            "success",
        )
    return (
        "Gemini Inactive",
        llm_status.get("message", "No Gemini status available."),
        "warning",
    )


def _render_insight_cards(items, columns=3):
    cols = st.columns(columns)
    for index, item in enumerate(items):
        with cols[index % columns]:
            st.markdown(f"**{item['title']}**")
            st.metric(item["value_label"], item["value"])
            st.caption(item["note"])


def _render_key_value_groups(groups):
    cols = st.columns(len(groups))
    for col, group in zip(cols, groups):
        with col:
            st.markdown(f"**{group['title']}**")
            for item in group["items"]:
                st.markdown(f"`{item['label']}` {item['value']}")


def _render_notification_summary(dispatch_frame):
    if dispatch_frame.empty:
        st.info("No notification deliveries have been recorded yet.")
        return
    latest = dispatch_frame.sort_values("Latest", ascending=False).head(3)
    for _, row in latest.iterrows():
        st.markdown(f"**{row['Alert']}**")
        st.caption(f"{row['Event']} · {row['Channels']} · {row['Level'].upper()}")
        st.write(row["Message"])


def _agent_summary_cards(latest_result, goal, event_context):
    critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
    planner_action = latest_result.get("planner_output", {}).get("proposed_action", {}).get("action", "none").upper()
    final_action = latest_result.get("action", {}).get("action", "none").upper()
    reward_score = latest_result.get("reward_score", 0)
    cards = [
        ("Planner", planner_action, latest_result.get("strategy", event_context.get("allocation_strategy", "-"))),
        ("Critic", latest_result.get("critic_output", {}).get("risk_level", "low").upper(), critic_notes[0] if critic_notes else "No critic issue raised."),
        ("Executor", final_action, latest_result.get("execution_output", {}).get("execution_note", "No execution note available.")),
        ("Goal", str(goal.get("target_congested_zones", "-")), goal.get("objective", "No active goal.")),
        ("Reward", f"{reward_score:+.2f}", "Positive means the latest transition improved network balance."),
    ]
    cols = st.columns(len(cards))
    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.metric(label, value)
            st.caption(note)


def _memory_summary_cards(metrics):
    cards = [
        ("Saved Steps", metrics.get("steps", 0)),
        ("Avg Search Time", metrics.get("avg_search_time_min", 0)),
        ("Avg Utilisation", metrics.get("avg_space_utilisation_pct", 0)),
        ("Avg Reward", metrics.get("avg_reward_score", 0)),
        ("Goal Updates", metrics.get("goal_updates", 0)),
    ]
    cols = st.columns(len(cards))
    for col, (label, value) in zip(cols, cards):
        with col:
            st.metric(label, value)


def _group_notification_dispatch(dispatch_rows):
    grouped = {}
    for row in dispatch_rows:
        key = (row.get("event"), row.get("title"), row.get("message"), row.get("level"))
        grouped.setdefault(key, {"channels": [], "latest_timestamp": row.get("timestamp", "")})
        grouped[key]["channels"].append(row.get("channel"))
        grouped[key]["latest_timestamp"] = max(grouped[key]["latest_timestamp"], row.get("timestamp", ""))
    records = []
    for (event, title, message, level), data in grouped.items():
        records.append(
            {
                "Event": event,
                "Alert": title,
                "Channels": ", ".join(sorted(set(data["channels"]))),
                "Level": level,
                "Latest": data["latest_timestamp"],
                "Message": message,
            }
        )
    return pd.DataFrame(records)


def _styled_state_frame(state_frame):
    return state_frame.style.format(
        {
            "Capacity": "{:.0f}",
            "Occupied": "{:.0f}",
            "Free": "{:.0f}",
            "Entries": "{:.0f}",
            "Exits": "{:.0f}",
            "Utilisation %": "{:.1f}",
        }
    ).background_gradient(subset=["Utilisation %"], cmap="RdYlGn_r")


def _styled_transition_frame(frame):
    if frame.empty:
        return frame
    return frame.style.format({"Before": "{:.0f}", "After": "{:.0f}", "Entries": "{:.0f}", "Exits": "{:.0f}", "Net Change": "{:+.0f}"}).background_gradient(
        subset=["Net Change"], cmap="RdYlGn"
    )


def _styled_cycle_frame(frame):
    if frame.empty:
        return frame
    return frame.style.format({"Reward": "{:+.2f}"}).background_gradient(subset=["Reward"], cmap="RdYlGn")


def main():
    st.set_page_config(layout="wide", page_title="SRM Agentic Parking Command Center")
    _inject_styles()
    _ensure_session_state()

    st.markdown(
        """
        <div class="hero-title">SRM Agentic Parking Command Center</div>
        <div class="hero-sub">
            Event-aware campus parking simulation with demand prediction, dynamic space allocation,
            measurable performance outcomes, and proactive parking recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    snapshot = runtime_service.get_runtime_snapshot()
    current_scenario = snapshot.get("scenario_mode", "Auto Schedule")

    st.sidebar.header("Simulation Controls")
    selected_scenario = st.sidebar.selectbox(
        "Campus Scenario",
        SCENARIOS,
        index=SCENARIOS.index(current_scenario) if current_scenario in SCENARIOS else 0,
    )
    if selected_scenario != current_scenario:
        runtime_service.set_scenario_mode(selected_scenario)
        st.rerun()

    st.session_state.run = st.sidebar.toggle("Autonomous Mode", value=st.session_state.run)
    speed = st.sidebar.slider("Step Interval (seconds)", 1.0, 8.0, 3.0)
    st.sidebar.caption(
        "Use Run One Step first. Switch scenarios to demonstrate campus events, agent decisions, and KPI changes without real-time data."
    )
    benchmark_episodes = st.sidebar.slider("Benchmark Episodes", 1, 5, 3)
    benchmark_steps = st.sidebar.slider("Benchmark Steps", 6, 15, 10)
    controls1, controls2 = st.sidebar.columns(2)
    with controls1:
        if st.button("Run One Step", width="stretch"):
            runtime_service.step()
            st.rerun()
    with controls2:
        if st.button("Pause", width="stretch"):
            st.session_state.run = False
            st.rerun()

    controls3, controls4 = st.sidebar.columns(2)
    with controls3:
        if st.button("Resume", width="stretch"):
            st.session_state.run = True
            st.rerun()
    with controls4:
        if st.button("Reset Runtime", width="stretch"):
            runtime_service.reset(clear_memory=False)
            st.session_state.run = False
            st.rerun()

    if st.sidebar.button("Reset Runtime + Memory", width="stretch"):
        runtime_service.reset(clear_memory=True)
        st.session_state.run = False
        st.rerun()
    if st.sidebar.button("Run Benchmark", width="stretch"):
        runtime_service.run_benchmark(
            episodes=benchmark_episodes,
            steps_per_episode=benchmark_steps,
        )
        st.session_state.run = False
        st.rerun()

    now = time.time()
    if st.session_state.run and now - st.session_state.last_run >= speed:
        st.session_state.last_run = now
        runtime_service.step()

    snapshot = runtime_service.get_runtime_snapshot()
    state = snapshot["state"]
    latest_result = snapshot.get("latest_result", {})
    latest_transition = snapshot.get("latest_transition", {})
    metrics = snapshot.get("metrics", {})
    goal = snapshot.get("goal", {})
    event_context = snapshot.get("event_context", {})
    kpis = snapshot.get("kpis", {})
    notifications = snapshot.get("notifications", [])
    notification_dispatch = snapshot.get("notification_dispatch", [])
    recent_cycles = snapshot.get("recent_cycles", [])
    recent_states = snapshot.get("recent_states", [])
    trace = snapshot.get("trace", [])
    llm_status = snapshot.get("llm_status", {})
    benchmark = snapshot.get("benchmark", {})
    operational_signals = latest_result.get("operational_signals", latest_transition.get("dynamic_signals", {}))
    assistant_briefing = snapshot.get("assistant_briefing", {})

    if llm_status.get("available"):
        st.sidebar.success(f"Gemini active: {llm_status.get('model', 'gemini')}")
        st.sidebar.caption("Planner, critic, and chat will try Gemini first.")
    else:
        st.sidebar.warning(f"Gemini inactive: {llm_status.get('message', 'No LLM status available.')}")

    state_frame = _format_state_table(state)
    total_capacity = int(state_frame["Capacity"].sum())
    total_occupied = int(state_frame["Occupied"].sum())
    total_free = int(state_frame["Free"].sum())
    congestion = round((total_occupied / total_capacity * 100), 2) if total_capacity else 0.0

    st.markdown(
        f"""
        <div class="event-banner">
            <div class="event-title">{event_context.get("name", "Campus Simulation")} | {event_context.get("time_window", "")}</div>
            <div class="event-copy">
                {event_context.get("description", "")}
                Strategy: <strong>{event_context.get("allocation_strategy", "Balanced utilisation")}</strong>.
                Recommended zone for incoming vehicles: <strong>{event_context.get("recommended_zone", "-")}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Capacity", total_capacity)
    m2.metric("Occupied", total_occupied)
    m3.metric("Free Slots", total_free)
    m4.metric("Congestion %", congestion)
    m5.metric("Saved Steps", metrics.get("steps", 0))

    cards = _signal_cards(event_context, latest_result, kpis, goal)
    st.markdown(
        "<div class='signal-grid'>"
        + "".join(
            f"<div class='signal-card'><div class='signal-label'>{card['label']}</div>"
            f"<div class='signal-value'>{card['value']}</div>"
            f"<div class='signal-note'>{card['note']}</div></div>"
            for card in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.run:
        st.info(f"Autonomous mode is active. The simulation refreshes every {speed:.1f} seconds.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Operations", "Events & KPIs", "Benchmark", "Agent Loop", "Memory & Goals", "Notifications", "AI Chat"]
    )

    with tab1:
        st.markdown("<div class='section-kicker'>Live Parking Network</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This section shows current space availability, simulated car movements, and the latest dynamic allocation result.</div>",
            unsafe_allow_html=True,
        )
        _render_story_cards(goal, latest_result, event_context, kpis)
        llm_title, llm_body, llm_level = _build_llm_summary(llm_status)
        _render_status_callout(llm_title, llm_body, llm_level)
        _render_assistant_briefing(assistant_briefing)
        _render_zone_cards(state_frame)

        left, right = st.columns([1, 1])
        left.plotly_chart(_build_zone_chart(state_frame), use_container_width=True, config={"displayModeBar": False})
        right.plotly_chart(_build_utilisation_chart(state_frame), use_container_width=True, config={"displayModeBar": False})

        with st.expander("Open detailed zone table", expanded=False):
            st.dataframe(
                _styled_state_frame(state_frame),
                width="stretch",
                hide_index=True,
                column_config={
                    "Utilisation %": st.column_config.ProgressColumn(
                        "Utilisation %",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    )
                },
            )

        flow_chart = _build_flow_chart(recent_states)
        if flow_chart is not None:
            st.plotly_chart(flow_chart, use_container_width=True, config={"displayModeBar": False})

        lower_left, lower_right = st.columns([1, 1])
        lower_left.markdown("<div class='section-kicker'>Latest Zone Transition</div>", unsafe_allow_html=True)
        lower_left.dataframe(_styled_transition_frame(_transition_frame(latest_transition)), width="stretch", hide_index=True)
        with lower_right:
            st.markdown("<div class='section-kicker'>Latest Allocation Summary</div>", unsafe_allow_html=True)
            _render_key_value_groups(
                [
                    {
                        "title": "Allocation Result",
                        "items": [
                            {"label": "Transferred vehicles", "value": latest_transition.get("transferred", 0)},
                            {"label": "Denied first-choice entries", "value": latest_transition.get("totals", {}).get("denied_entries", 0)},
                        ],
                    },
                    {
                        "title": "Routing Context",
                        "items": [
                            {"label": "Recommended zone", "value": event_context.get("recommended_zone", "-")},
                            {"label": "Focus zone", "value": event_context.get("focus_zone", "-")},
                        ],
                    },
                ]
            )

    with tab2:
        st.markdown("<div class='section-kicker'>Industry Outcome Metrics</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>These are the measurable business outcomes this simulation is designed to improve: utilisation, search time, congestion hotspots, and allocation success.</div>",
            unsafe_allow_html=True,
        )
        _render_html_block(
            "<div class='feature-callout'><strong>Presentation view:</strong> this dashboard now makes the impact of the agent visible in plain language through KPIs, routing strategy, and student-facing notifications instead of only raw tables.</div>"
        )
        _render_insight_cards(
            [
                {"title": "Search Time", "value_label": "Estimated", "value": f"{kpis.get('estimated_search_time_min', 0)} min", "note": "Lower is better."},
                {"title": "Space Utilisation", "value_label": "Current", "value": f"{kpis.get('space_utilisation_pct', 0)}%", "note": "Healthy utilisation without overloading."},
                {"title": "Allocation Success", "value_label": "Current", "value": f"{kpis.get('allocation_success_pct', 0)}%", "note": "Share of successful dynamic reallocations."},
                {"title": "Congestion Hotspots", "value_label": "Current", "value": str(kpis.get('congestion_hotspots', 0)), "note": "Zones with critically low free space."},
                {"title": "Balance Index", "value_label": "Current", "value": str(kpis.get('balance_index', 0)), "note": "Lower means better load balancing."},
            ],
            columns=5,
        )

        kpi_chart = _build_kpi_chart(recent_states)
        if kpi_chart is not None:
            st.plotly_chart(kpi_chart, use_container_width=True, config={"displayModeBar": False})

        _render_key_value_groups(
            [
                {
                    "title": "Driver Guidance",
                    "items": [
                        {"label": "Advisory", "value": event_context.get("user_advisory", "-")},
                    ],
                },
                {
                    "title": "Routing Strategy",
                    "items": [
                        {"label": "Active strategy", "value": event_context.get("allocation_strategy", "-")},
                    ],
                },
                {
                    "title": "Best Overflow Zone",
                    "items": [
                        {"label": "Recommended zone", "value": event_context.get("recommended_zone", "-")},
                    ],
                },
            ]
        )

    with tab3:
        st.markdown("<div class='section-kicker'>Simulation Proof</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This benchmark compares the current agentic runtime against a no-redirect baseline over the same synthetic scenarios and seeds.</div>",
            unsafe_allow_html=True,
        )
        aggregate = benchmark.get("aggregate", {})
        _render_insight_cards(
            [
                {"title": "Search Time Gain", "value_label": "Average", "value": f"{aggregate.get('avg_search_time_gain_min', 0)} min", "note": "Positive means agentic mode reduces average search time."},
                {"title": "Resilience Gain", "value_label": "Average", "value": aggregate.get("avg_resilience_gain", 0), "note": "Higher is better under synthetic disruptions."},
                {"title": "Hotspot Reduction", "value_label": "Average", "value": aggregate.get("avg_hotspot_reduction", 0), "note": "Positive means fewer critically congested zones."},
            ],
            columns=3,
        )
        benchmark_frame = _build_benchmark_frame(benchmark)
        chart = _build_benchmark_chart(benchmark_frame)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(benchmark_frame, width="stretch", hide_index=True)
        else:
            st.info("Run the benchmark from the sidebar to generate a baseline comparison.")

    with tab4:
        st.markdown("<div class='section-kicker'>Planner, Critic, Executor</div>", unsafe_allow_html=True)
        _agent_summary_cards(latest_result, goal, event_context)
        left, right = st.columns([1.05, 1])
        agent_df = _agent_frame(latest_result)
        if not agent_df.empty:
            left.dataframe(agent_df[["Agent", "Action Taken"]], width="stretch", hide_index=True)
        else:
            left.info("Run the system to populate the agent loop.")

        with right:
            critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
            final_action_text = latest_result.get("action", {}).get("reason")
            if not final_action_text:
                final_action_text = critic_notes[0] if critic_notes else "The agent held the current allocation because the network is stable enough for this step."
            _render_key_value_groups(
                [
                    {
                        "title": "Decision Summary",
                        "items": [
                            {"label": "Planner strategy", "value": latest_result.get("strategy", event_context.get("allocation_strategy", "-"))},
                            {"label": "Final action", "value": latest_result.get("action", {}).get("action", "none").upper()},
                        ],
                    },
                    {
                        "title": "Reasoning",
                        "items": [
                            {"label": "Goal", "value": goal.get("objective", "No goal yet")},
                            {"label": "Critic outcome", "value": "; ".join(critic_notes) or final_action_text},
                        ],
                    },
                ]
            )
            st.caption(final_action_text)
            if latest_result.get("autonomy", {}).get("replan_triggered"):
                st.warning("Autonomy monitor triggered a replan because current KPI pressure exceeded the goal threshold.")
            with st.expander("Open raw planner / critic / executor payload"):
                st.json(
                    {
                        "planner_output": latest_result.get("planner_output", {}),
                        "critic_output": latest_result.get("critic_output", {}),
                        "execution_output": latest_result.get("execution_output", {}),
                        "policy_baseline": latest_result.get("policy_action", {}),
                        "autonomy": latest_result.get("autonomy", {}),
                    }
                )

        cycle_df = _cycle_frame(recent_cycles)
        if not cycle_df.empty:
            with st.expander("Open recent agent cycle history", expanded=False):
                st.dataframe(_styled_cycle_frame(cycle_df.tail(6)), width="stretch", hide_index=True)

    with tab5:
        st.markdown("<div class='section-kicker'>Persistent Learning View</div>", unsafe_allow_html=True)
        _memory_summary_cards(metrics)
        _render_key_value_groups(
            [
                {
                    "title": "Learning Snapshot",
                    "items": [
                        {"label": "Average free slots", "value": metrics.get("avg_free_slots", 0)},
                        {"label": "Average allocation success", "value": f"{metrics.get('allocation_success_pct', 0)}%"},
                        {"label": "Learning status", "value": "Adaptive reward + Q-learning"},
                        {"label": "Recent failures tracked", "value": len(metrics.get("learning_profile", {}).get("recent_failures", []))},
                    ],
                },
                {
                    "title": "Active Goal",
                    "items": [
                        {"label": "Objective", "value": goal.get("objective", "No goal yet")},
                        {"label": "Priority zone", "value": goal.get("priority_zone", "-")},
                        {"label": "Target congested zones", "value": goal.get("target_congested_zones", "-")},
                        {"label": "Target search time", "value": goal.get("target_search_time_min", "-")},
                        {"label": "Horizon steps", "value": goal.get("horizon_steps", "-")},
                    ],
                },
            ]
        )
        if operational_signals:
            _render_key_value_groups(
                [
                    {
                        "title": "Synthetic Sensor Signals",
                        "items": [
                            {"label": "Weather", "value": operational_signals.get("weather", "Clear")},
                            {"label": "Queue length", "value": operational_signals.get("queue_length", 0)},
                            {"label": "Blocked zone", "value": operational_signals.get("blocked_zone", "-") or "-"},
                            {"label": "Patrol mode", "value": operational_signals.get("patrol_mode", "Normal")},
                        ],
                    }
                ]
            )

        trace_frame = _trace_frame(trace)
        if not trace_frame.empty:
            with st.expander("Open memory trace log", expanded=False):
                st.dataframe(trace_frame.tail(8), width="stretch", hide_index=True)

    with tab6:
        st.markdown("<div class='section-kicker'>Proactive User Notifications</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>This simulates how the platform would notify drivers before congestion becomes severe, even without live campus sensor feeds.</div>",
            unsafe_allow_html=True,
        )
        _render_notifications(notifications)
        if notification_dispatch:
            st.markdown("<div class='section-kicker'>Mock Delivery Channels</div>", unsafe_allow_html=True)
            dispatch_frame = _group_notification_dispatch(notification_dispatch)
            _render_notification_summary(dispatch_frame)
            with st.expander("Open delivery audit log", expanded=False):
                st.dataframe(dispatch_frame, width="stretch", hide_index=True)

    with tab7:
        st.markdown("<div class='section-kicker'>Parking Operations Assistant</div>", unsafe_allow_html=True)
        _render_assistant_briefing(assistant_briefing)
        suggestion_cols = st.columns(4)
        suggestions = [
            "Which zone is best right now?",
            "What event is affecting parking?",
            "Show the latest allocation decision",
            "Which block is most congested?",
        ]
        for col, suggestion in zip(suggestion_cols, suggestions):
            with col:
                if st.button(suggestion, width="stretch"):
                    st.session_state.chat_response = runtime_service.ask(suggestion)["answer"]
        with st.form("chat_form", clear_on_submit=False):
            query = st.text_input("Ask about parking rush, best zones, current event impact, or dynamic allocation")
            submitted = st.form_submit_button("Ask")

        if submitted and query.strip():
            st.session_state.chat_response = runtime_service.ask(query.strip())["answer"]

        if st.session_state.chat_response:
            st.success(st.session_state.chat_response)

    if st.session_state.run:
        _schedule_reload(speed)


if __name__ == "__main__":
    main()
