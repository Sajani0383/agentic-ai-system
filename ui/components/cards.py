import streamlit as st
from ui.config import PALETTE

def render_html_block(html):
    st.markdown(html, unsafe_allow_html=True)

def signal_cards(event_context, latest_result, kpis, goal):
    action = latest_result.get("action", {})
    critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
    action_reason = action.get("reason")
    if not action_reason:
        action_reason = critic_notes[0] if critic_notes else "The agent is holding position because current pressure does not justify a transfer."
    
    cards = [
        {"label": "Current Event", "value": event_context.get("name", "Normal Day"), "note": f"Severity: {event_context.get('severity', 'low').title()} | Focus: {event_context.get('focus_zone', '-')}"},
        {"label": "Allocation Strategy", "value": latest_result.get("strategy", event_context.get("allocation_strategy", "Balanced utilisation")), "note": f"Recommended overflow: {event_context.get('recommended_zone', '-')}"},
        {"label": "Live Decision", "value": f"{action.get('action', 'none').upper()} | {latest_result.get('mode', 'idle').replace('_', ' ')}", "note": action_reason},
        {"label": "Measured Outcome", "value": f"{kpis.get('estimated_search_time_min', 0)} min search time", "note": f"Avg Utilisation: {kpis.get('space_utilisation_pct', 0)}%"},
    ]
    
    render_html_block("<div class='signal-grid'>" + "".join(
        f"<div class='signal-card'><div class='signal-label'>{c['label']}</div><div class='signal-value'>{c['value']}</div><div class='signal-note'>{c['note']}</div></div>"
        for c in cards
    ) + "</div>")

def render_story_cards(goal, latest_result, event_context, kpis):
    action = latest_result.get("action", {})
    story_cards = [
        {"title": "Why The Agent Acted", "value": event_context.get("focus_zone", "Campus"), "copy": action.get("reason", "The agent is waiting for enough demand pressure to justify reallocation.")},
        {"title": "Active Goal", "value": f"{goal.get('target_congested_zones', '-') } zone target", "copy": goal.get("objective", "No active goal yet.")},
        {"title": "Student Outcome", "value": f"{kpis.get('estimated_search_time_min', 0)} min", "copy": "Estimated parking search time after the current allocation decision."},
    ]
    render_html_block("<div class='story-grid'>" + "".join(
        f"<div class='story-card'><div class='story-title'>{c['title']}</div><div class='story-value'>{c['value']}</div><div class='story-copy'>{c['copy']}</div></div>"
        for c in story_cards
    ) + "</div>")

def render_insight_cards(items, columns=3):
    cols = st.columns(columns)
    for index, item in enumerate(items):
        with cols[index % columns]:
            st.markdown(f"**{item['title']}**")
            st.metric(item["value_label"], item["value"])
            st.caption(item["note"])

def recommendation_tone(label):
    if label == "Avoid": return "coral"
    if label == "Overflow": return "gold"
    return "green"

def render_zone_cards(state_frame):
    zone_columns = st.columns(len(state_frame))
    for column, (_, row) in zip(zone_columns, state_frame.iterrows()):
        utilisation = float(row["Utilisation %"])
        tone = recommendation_tone(row["Recommendation"])
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

def render_key_value_groups(groups):
    cols = st.columns(len(groups))
    for col, group in zip(cols, groups):
        with col:
            st.markdown(f"**{group['title']}**")
            for item in group["items"]:
                st.markdown(f"`{item['label']}` {item['value']}")
