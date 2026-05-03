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
    local_target = action.get("to") or "-"
    local_source = action.get("from") or "-"
    pressure_focus = event_context.get("pressure_focus") or event_context.get("focus_zone", "-")
    source_note = f" | Action source: {local_source}" if local_source != "-" else ""
    action_name = str(action.get("action", "none")).upper()
    if action_name == "NONE":
        action_name = "IDLE"
    if action_name not in {"REDIRECT", "REPLAN", "IDLE"}:
        action_name = "REDIRECT" if action.get("from") and action.get("to") else "IDLE"
    vehicle_ids = action.get("vehicle_ids") or []
    vehicle_note = f" | Vehicles: {', '.join(map(str, vehicle_ids[:5]))}" if vehicle_ids else ""
    reward_score = float(latest_result.get("reward_score", 0) or 0)
    search_delta = latest_result.get("baseline_comparison", {}).get("search_time_delta_min")
    reward_note = ""
    if reward_score < -0.05 and search_delta and float(search_delta or 0) > 0:
        reward_note = " Reward penalized execution cost despite search improvement."
    recommendation_note = (
        f"Recommended (global): {event_context.get('recommended_zone', '-')} | "
        f"Chosen (local): {local_target}. Global handles incoming flow; local resolves active congestion."
    )
    perception_note = (
        f"Focus = pressure source: {pressure_focus}. Source = rerouting origin: {local_source}."
        if local_source != "-" else f"Focus = pressure source: {pressure_focus}."
    )
    
    cards = [
        {"label": "SRM Scenario", "value": event_context.get("name", "Normal Day"), "note": f"Severity: {event_context.get('severity', 'low').title()} | Pressure focus: {pressure_focus}{source_note}"},
        {"label": "Allocation Strategy", "value": latest_result.get("strategy", event_context.get("allocation_strategy", "Balanced utilisation")), "note": recommendation_note},
        {"label": "Live Decision", "value": action_name, "note": f"{action_reason}{vehicle_note}"},
        {"label": "Decision Link", "value": f"{local_source} → {local_target}" if local_source != "-" and local_target != "-" else "Monitoring", "note": perception_note},
        {"label": "Measured Outcome", "value": f"{kpis.get('estimated_search_time_min', 0)} min search time", "note": f"Avg Utilisation: {kpis.get('space_utilisation_pct', 0)}%.{reward_note}"},
    ]
    
    render_html_block("<div class='signal-grid'>" + "".join(
        f"<div class='signal-card'><div class='signal-label'>{c['label']}</div><div class='signal-value'>{c['value']}</div><div class='signal-note'>{c['note']}</div></div>"
        for c in cards
    ) + "</div>")

def render_story_cards(goal, latest_result, event_context, kpis):
    action = latest_result.get("action", {})
    target_blocks = goal.get("target_congested_zones")
    goal_value = "Maintain balanced load"
    if target_blocks not in (None, "", "-"):
        try:
            target_count = int(target_blocks)
        except (TypeError, ValueError):
            target_count = None
        goal_value = "Maintain low congestion" if target_count == 0 else f"{target_blocks} hotspot target"
    story_cards = [
        {"title": "Why The Agent Acted", "value": event_context.get("focus_zone", "SRM"), "copy": action.get("reason", "The agent is waiting for enough SRM block pressure to justify reallocation.")},
        {"title": "Active Goal", "value": goal_value, "copy": goal.get("objective", "No active goal yet.")},
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
    rows = list(state_frame.iterrows())
    for start in range(0, len(rows), 4):
        zone_columns = st.columns(min(4, len(rows) - start))
        for column, (_, row) in zip(zone_columns, rows[start:start + 4]):
            utilisation = float(row["Utilisation %"])
            tone = recommendation_tone(row["Recommendation"])
            with column:
                st.markdown(f"**{row['Zone']}**")
                st.caption(f"Car {int(row.get('Car Slots', 0))} · Bike {int(row.get('Bike Slots', 0))} · Total {int(row['Capacity'])}")
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
