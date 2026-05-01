import json
import time
import pandas as pd
import streamlit as st

from ui.config import SCENARIOS, STRINGS
from ui.components.styles import inject_styles
from ui.logic.api_bridge import api_bridge
from ui.logic.state_manager import state_manager
from ui.logic.input_validator import validator

from ui.components.charts import (
    build_zone_chart,
    build_utilisation_chart,
    build_flow_chart,
    build_kpi_chart,
    build_benchmark_chart,
    build_latest_baseline_chart,
    build_performance_trend_chart,
)
from ui.components.cards import (
    render_html_block,
    signal_cards,
    render_story_cards,
    render_insight_cards,
    render_zone_cards,
    render_key_value_groups,
)
from ui.components.tables import (
    styled_state_frame,
    styled_transition_frame,
    styled_cycle_frame,
)

def _ensure_session_state():
    if "run" not in st.session_state:
        st.session_state.run = False
    if "last_run" not in st.session_state:
        st.session_state.last_run = 0.0
    if "last_ui_interaction_at" not in st.session_state:
        st.session_state.last_ui_interaction_at = 0.0
    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""
    if "chat_response_meta" not in st.session_state:
        st.session_state.chat_response_meta = {}
    if "benchmark_toggle" not in st.session_state:
        st.session_state.benchmark_toggle = False
    if "force_llm" not in st.session_state:
        st.session_state.force_llm = False
    if "run_report_json" not in st.session_state:
        st.session_state.run_report_json = ""

def _mark_ui_interaction():
    st.session_state.last_ui_interaction_at = time.time()

def _ui_recently_interacted(now=None, cooldown=0.75):
    now = time.time() if now is None else now
    last_interaction = float(st.session_state.get("last_ui_interaction_at", 0.0) or 0.0)
    return (now - last_interaction) < cooldown

def _schedule_reload(seconds):
    time.sleep(max(0.2, seconds))
    st.rerun()

def _build_llm_summary(llm_status, is_forced=False):
    quota_backoff = llm_status.get("quota_backoff", {})
    if is_forced:
        return (
            "⚡ Strategic Overdrive Active",
        "System is strictly prioritizing live Gemini reasoning for SRM parking operations. Safety locks and quota cooldowns are currently overridden.",
            "success",
        )
    if quota_backoff.get("active"):
        if quota_backoff.get("kind") == "daily_quota":
            return (
                "Live Gemini Paused By Daily Quota",
                "The active Gemini project/key has reached its free-tier daily cap. The planner is staying in simulated and local agentic reasoning so the parking system keeps operating without blank steps or broken execution.",
                "info",
            )
        return (
            "Efficient Local Mode Active",
            f"System has transitioned to Autonomous Edge Intelligence ({quota_backoff.get('remaining_seconds', 0)}s remaining in cloud optimization cycle). Resilient multi-agent heuristics are currently driving SRM parking allocation.",
            "info",
        )
    if llm_status.get("available"):
        message = llm_status.get("message", "Gemini SDK is configured.")
        if llm_status.get("last_error"):
            return (
                "Hybrid Intelligence Active",
                f"Model: {llm_status.get('model', 'gemini')}. {message} Local reasoning is currently prioritizing edge-based optimization.",
                "info",
            )
        return (
            "Full Agentic Intelligence Ready",
            f"Model: {llm_status.get('model', 'gemini')}. System is running in Hybrid Mode, balancing cloud-based LLM reasoning with efficient SRM parking heuristics.",
            "success",
        )
    return (
        "Autonomous Edge Mode",
        "System is operating in full local-agent mode. Cloud intelligence is currently restricted or disabled.",
        "info",
    )

def _render_assistant_briefing(briefing):
    if not briefing:
        return
    st.markdown("**SRM Parking AI Copilot**")
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

def _render_notifications(notifications):
    if not notifications:
        st.info("No active SRM parking alerts right now.")
        return
    for notification in notifications:
        text = f"**{notification.get('title', 'Update')}**  \n{notification.get('message', '')}"
        level = notification.get("level")
        if level == "error": st.error(text)
        elif level == "warning": st.warning(text)
        elif level == "success": st.success(text)
        else: st.info(text)

def _render_srm_parking_overview(state_frame):
    if state_frame.empty:
        return
    car_slots = int(state_frame.get("Car Slots", pd.Series(dtype=int)).sum())
    bike_slots = int(state_frame.get("Bike Slots", pd.Series(dtype=int)).sum())
    total_capacity = int(state_frame["Capacity"].sum())
    highest_pressure = state_frame.sort_values("Utilisation %", ascending=False).head(1)
    best_buffer = state_frame.sort_values("Free", ascending=False).head(1)
    pressure_block = highest_pressure.iloc[0]["Zone"] if not highest_pressure.empty else "-"
    buffer_block = best_buffer.iloc[0]["Zone"] if not best_buffer.empty else "-"
    render_key_value_groups([
        {
            "title": "SRM Capacity Map",
            "items": [
                {"label": "Parking blocks", "value": len(state_frame)},
                {"label": "Car slots", "value": car_slots},
                {"label": "Bike slots", "value": bike_slots},
                {"label": "Total capacity", "value": total_capacity},
            ],
        },
        {
            "title": "Live Block Focus",
            "items": [
                {"label": "Highest pressure", "value": pressure_block},
                {"label": "Best buffer", "value": buffer_block},
                {"label": "Free at buffer", "value": int(best_buffer.iloc[0]["Free"]) if not best_buffer.empty else "-"},
                {"label": "Capacity source", "value": "SRM block dataset"},
            ],
        },
    ])

def _agent_summary_cards(latest_result, goal, event_context):
    critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
    planner_action = latest_result.get("planner_output", {}).get("proposed_action", {}).get("action", "none").upper()
    final_action = latest_result.get("action", {}).get("action", "none").upper()
    reward_score = latest_result.get("reward_score", 0)
    cards = [
        ("Planner", planner_action, latest_result.get("strategy", event_context.get("allocation_strategy", "-"))),
        ("Critic", latest_result.get("critic_output", {}).get("risk_level", "low").upper(), critic_notes[0] if critic_notes else "No critic issue raised."),
        ("Executor", final_action, _execution_summary(latest_result)),
        ("Goal", str(goal.get("target_congested_zones", "-")), goal.get("objective", "No active goal.")),
        ("Reward", f"{reward_score:+.2f}", _reward_summary(latest_result)),
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

def _reward_summary(latest_result):
    reward_score = float(latest_result.get("reward_score", 0) or 0)
    reward_impact = _safe_dict(latest_result.get("reward_impact"))
    if reward_score < -0.05:
        return reward_impact.get("explanation", "Negative reward: route is penalized and repeated failures are blocked.")
    if reward_score > 0.05:
        return reward_impact.get("explanation", "Positive reward: route confidence reinforced.")
    return "Neutral reward: no major route weight change."

def _render_goal_status(goal, kpis):
    if not goal:
        st.info("No active goal has been created yet. Run one step to let the planner set a goal.")
        return
    target_hotspots = goal.get("target_congested_zones", 1)
    target_search = goal.get("target_search_time_min", 4.0)
    current_hotspots = kpis.get("congestion_hotspots", 0)
    current_search = kpis.get("estimated_search_time_min", 0.0)
    achieved = current_hotspots <= target_hotspots and current_search <= target_search
    status = "Achieved" if achieved else "In Progress"
    render_key_value_groups([
        {
            "title": "SRM Goal Status",
            "items": [
                {"label": "Status", "value": status},
                {"label": "Objective", "value": goal.get("objective", "-")},
                {"label": "Priority block", "value": goal.get("priority_zone", "-")},
            ],
        },
        {
            "title": "Target Check",
            "items": [
                {"label": "Hotspots", "value": f"{current_hotspots} / target {target_hotspots}"},
                {"label": "Search time", "value": f"{current_search} / target {target_search} min"},
                {"label": "Horizon", "value": goal.get("horizon_steps", "-")},
            ],
        },
    ])
    if achieved:
        st.success("The current SRM parking step satisfies the active goal thresholds.")
    else:
        st.warning("The goal is still active; the planner will keep monitoring pressure and route options.")

def _render_reasoning_budget(reasoning_budget):
    if not reasoning_budget:
        st.info("No reasoning budget has been computed yet.")
        return

    signals = reasoning_budget.get("signals", {})
    top_cols = st.columns(4)
    top_cols[0].metric("Budget Mode", reasoning_budget.get("budget_level", "local_only").replace("_", " ").title())
    top_cols[1].metric("Queue", signals.get("queue_length", 0))
    top_cols[2].metric("Entropy", signals.get("entropy", 0))
    top_cols[3].metric("Hotspots", signals.get("hotspots", 0))

    st.caption(reasoning_budget.get("planner_reason", ""))
    gate_notes = reasoning_budget.get("gate_notes", [])
    if gate_notes:
        st.warning(" ".join(gate_notes))
    st.caption(
        f"Cooldown remaining: {signals.get('cooldown_remaining', 0)} step(s) | "
        f"Demand delta: {signals.get('demand_delta', 0)} | "
        f"Free-slot delta: {signals.get('free_slot_delta', 0)} | "
        f"Quota backoff: {signals.get('quota_backoff_remaining_seconds', 0)} sec"
    )
    st.caption(
        f"LLM cadence: every {signals.get('llm_stride_steps', 15)} step(s) | "
        f"Next scheduled LLM step: {signals.get('next_scheduled_llm_step', 0)} | "
        f"Steps until next LLM: {signals.get('steps_until_next_llm', 0)}"
    )
    left, right = st.columns(2)
    with left:
        moderate = reasoning_budget.get("moderate_triggers", [])
        st.markdown("**Moderate triggers**")
        if moderate:
            for item in moderate:
                st.markdown(f"- {item}")
        else:
            st.write("None")
    with right:
        severe = reasoning_budget.get("severe_triggers", [])
        st.markdown("**Severe triggers**")
        if severe:
            for item in severe:
                st.markdown(f"- {item}")
        else:
            st.write("None")

def _get_llm_state(latest_result, llm_status, llm_mode):
    planner = latest_result.get("planner_output", {})
    critic = latest_result.get("critic_output", {})
    budget = latest_result.get("reasoning_budget", {})
    backoff = llm_status.get("quota_backoff", {})
    requested = bool(planner.get("llm_requested") or critic.get("llm_requested"))
    planned = bool(budget.get("allow_planner_llm") or budget.get("allow_critic_llm"))
    if backoff.get("active"):
        if backoff.get("kind") == "daily_quota":
            return {
                "mode": "Budget Guard",
                "status": "Daily Quota Exhausted",
                "fallback": "Simulated / Local Reasoning",
                "detail": "Gemini free-tier daily quota is exhausted for the active project/key. The system has intentionally switched to simulated and local agentic reasoning until quota resets or a new quota source is provided.",
            }
        if llm_mode == "demo":
            return {
                "mode": "Demo Requested",
                "status": "Gemini Cooldown",
                "fallback": "Simulated Gemini",
                "detail": f"Demo mode requested Gemini, but a {backoff.get('kind', 'backoff')} cooldown is active for {backoff.get('remaining_seconds', 0)} sec. Last error: {llm_status.get('last_error', 'n/a')}. Simulated Gemini remains available for demo continuity.",
            }
        return {
            "mode": "Auto Budget" if llm_mode == "auto" else "Local Only",
            "status": "Gemini Cooldown",
            "fallback": "Local Fallback",
            "detail": f"Provider {backoff.get('kind', 'backoff')} cooldown is active for {backoff.get('remaining_seconds', 0)} sec. Last error: {llm_status.get('last_error', 'n/a')}.",
        }
    if llm_mode == "local":
        return {"mode": "Local Only", "status": "Gemini Skipped", "fallback": "Deterministic", "detail": "Operator selected Local Only mode."}
    if requested:
        planner = latest_result.get("planner_output", {})
        source = planner.get("llm_source", "gemini")
        if planner.get("llm_fallback_used") or source == "gemini_failed_fallback":
            return {
                "mode": "Gemini Attempted",
                "status": "Fallback Active",
                "fallback": "Local Agentic Pipeline",
                "detail": planner.get("llm_fallback_reason") or planner.get("llm_error") or "Gemini was requested, but local deterministic agents completed the decision.",
            }
        if source == "demo_simulated":
            return {"mode": "Demo Simulated Gemini", "status": "Simulated Gemini", "fallback": "Not Needed", "detail": "Demo mode produced a simulated Gemini advisory because live Gemini was unavailable."}
        return {"mode": "Demo Planner LLM" if llm_mode == "demo" else "Auto Escalated", "status": "Gemini Used", "fallback": "Not Needed", "detail": "Gemini advisory was requested for this step."}
    if planned:
        return {"mode": "Gemini Planned", "status": "Fallback Active", "fallback": "Local Fallback", "detail": "The gate allowed Gemini, but the runtime completed with deterministic fallback."}
    if llm_mode == "demo":
        return {"mode": "Demo Requested", "status": "Gemini Unavailable", "fallback": "Local Fallback", "detail": llm_status.get("message", "Gemini was not available for this step.")}
    return {"mode": "Auto Budget", "status": "Local Reasoning", "fallback": "Deterministic", "detail": "No high-value LLM trigger was needed."}

def _render_system_status_bar(latest_result, event_context, kpis, llm_status, llm_mode):
    critic = latest_result.get("critic_output", {})
    action = latest_result.get("action", {})
    confidence = action.get("confidence", latest_result.get("planner_output", {}).get("confidence", "-"))
    if isinstance(confidence, float):
        confidence = f"{confidence:.0%}" if confidence <= 1 else f"{confidence:.0f}%"
    llm_state = _get_llm_state(latest_result, llm_status, llm_mode)
    render_html_block(
        f"""
        <div class="status-bar">
            <div><span>Mode</span><strong>{llm_state["mode"]}</strong></div>
            <div><span>LLM</span><strong>{llm_state["status"]}</strong></div>
            <div><span>Fallback</span><strong>{llm_state["fallback"]}</strong></div>
            <div><span>Risk</span><strong>{critic.get("risk_level", "low").title()}</strong></div>
            <div><span title="Based on risk constraints, capacity bounds, and network demand">Confidence ℹ️</span><strong>{confidence}</strong></div>
        </div>
        <div class="status-note">{llm_state["detail"]} Search time: {kpis.get('estimated_search_time_min', 0):.1f} min. Hotspots: {kpis.get('congestion_hotspots', 0)}.</div>
        """
    )

def _render_decision_summary_block(latest_result, baseline_comparison, updated_at=""):
    if not latest_result:
        return
    baseline_comparison = _safe_dict(baseline_comparison)
    action = latest_result.get("action", {})
    act_type = action.get("action", "none").upper()
    impact = _format_decision_impact(baseline_comparison)
    if act_type == "REDIRECT":
        source = action.get("from", "Unknown")
        target = action.get("to", "Unknown")
        vehicles = int(action.get("vehicles", 0))
        punchy_reason = action.get("reason") or f"{source} is under pressure and {target} has the best free capacity, so the agent is redirecting {vehicles} vehicles right now."
        display_type = "REDIRECT"
        route = f"{source} → {target}"
    else:
        state = latest_result.get("state", {})
        if state:
            crowded = min(state, key=lambda z: state[z].get("free_slots", 0))
            best = max(state, key=lambda z: state[z].get("free_slots", 0))
            source = crowded
            target = best
        else:
            crowded, best, source, target = "-", "-", "-", "-"
        vehicles = 0
        punchy_reason = action.get("reason") or f"The agent reviewed all SRM blocks and kept monitoring active because {source} remains within safe limits for this cycle."
        display_type = "MONITORING"
        route = f"{source} stable | best buffer {target}"

    render_html_block(
        f"""
        <div class="realtime-badge"><span class="realtime-dot"></span>Live backend timestamp: {_format_live_timestamp(updated_at)}</div>
        <div class="focus-grid">
            <div class="focus-card">
                <div class="focus-kicker">Current Decision</div>
                <div class="focus-title">{display_type}</div>
                <div class="focus-route">{route}</div>
                <div class="focus-reason">{punchy_reason}</div>
                <div class="focus-stat-grid">
                    <div class="focus-stat"><span>Vehicles</span><strong>{vehicles}</strong></div>
                    <div class="focus-stat"><span>Source</span><strong>{source}</strong></div>
                    <div class="focus-stat"><span>Destination</span><strong>{target}</strong></div>
                </div>
            </div>
            <div class="focus-card accent">
                <div class="focus-kicker">Decision Impact</div>
                <div class="focus-impact">{impact}</div>
                <div class="focus-stat-grid">
                    <div class="focus-stat"><span>Search Delta</span><strong>{baseline_comparison.get('search_time_delta_min', 0):+.2f}m</strong></div>
                    <div class="focus-stat"><span>Hotspot Delta</span><strong>{baseline_comparison.get('hotspot_delta', 0):+.0f}</strong></div>
                    <div class="focus-stat"><span>Execution</span><strong>{_execution_summary(latest_result)}</strong></div>
                </div>
            </div>
        </div>
        """
    )

def _render_llm_insight(latest_result):
    """Always-visible LLM decision panel — shows what Gemini decided OR why it was skipped."""
    if not latest_result:
        return
    critic = latest_result.get("critic_output", {})
    planner = latest_result.get("planner_output", {})
    budget = latest_result.get("reasoning_budget", {})
    llm_advisory_used = critic.get("llm_advisory_used") or planner.get("llm_advisory_used") or planner.get("llm_requested")
    llm_fallback_used = planner.get("llm_fallback_used") or critic.get("llm_fallback_used") or planner.get("llm_source") == "gemini_failed_fallback"
    budget_level = budget.get("budget_level", "local_only")
    planner_reason = budget.get("planner_reason", "")
    rationale = planner.get("rationale") or (critic.get("critic_notes") or [""])[0] or ""
    gate_notes = budget.get("gate_notes", [])

    if llm_fallback_used:
        st.markdown(
            f"""
            <div style="background: rgba(255,184,77,0.08); border-left: 3px solid #ffb84d; padding: 0.65rem 1rem; border-radius: 4px 10px 10px 4px; margin-bottom: 1rem;">
                <strong style="color: #ffcf80; display: block; margin-bottom: 0.2rem;">Gemini Attempted - Local Fallback Executed</strong>
                <span style="font-size: 0.85rem; color: #d0d7e1; display: block;">{planner.get('llm_fallback_reason') or planner.get('llm_error') or 'Gemini was unavailable, so the deterministic agentic pipeline completed the decision.'}</span>
                <span style="font-size: 0.78rem; color: #8da0b7; margin-top: 0.2rem; display:block;">Final mode: <b>{planner.get('decision_mode', 'local_fallback')}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif llm_advisory_used:
        # LLM WAS used — show what it said prominently
        st.markdown(
            f"""
            <div class="llm-insight-box">
                <strong style="color: #9f8cff; display: block; margin-bottom: 0.3rem;">🤖 Gemini Advisory Active — LLM Decision This Step</strong>
                <span style="font-size: 0.95rem; color: #d0d7e1; display: block; margin-bottom: 0.4rem;">{rationale}</span>
                <span style="font-size: 0.78rem; color: #7a8fa6;">Budget level: <b>{budget_level}</b> | Mode: {planner.get('decision_mode', 'llm_advisory')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # LLM was NOT used — explain why in a compact visible note
        gate_msg = " ".join(gate_notes) if gate_notes else planner_reason
        action = latest_result.get("action", {})
        planner_source = planner.get("llm_source", "deterministic")
        action_desc = (
            f"Redirect {action.get('vehicles',0)} vehicles from {action.get('from','-')} → {action.get('to','-')}"
            if action.get("action") == "redirect"
            else "No redirect — system stable"
        )
        mode_title = "Reasoning Core Active"
        mode_text = "Local simulated reasoning is driving this step while live Gemini is unavailable."
        if planner_source == "cached":
            mode_text = "Cached Gemini reasoning is driving this step to preserve prior LLM behavior."
        elif planner_source == "deterministic":
            mode_title = "Deterministic Step"
            mode_text = "Deterministic agents handled this step because the state did not justify extra reasoning cost."
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.02); border-left: 3px solid rgba(150,150,180,0.4); padding: 0.65rem 1rem; border-radius: 4px 10px 10px 4px; margin-bottom: 1rem;">
                <strong style="color: #8899bb; display: block; margin-bottom: 0.2rem;">⚡ {mode_title}</strong>
                <span style="font-size: 0.85rem; color: #7a8fa6; display: block;">Decision: <b style='color:#ccd3d9'>{action_desc}</b></span>
                <span style="font-size: 0.78rem; color: #5e6e82; margin-top: 0.2rem; display:block;">{mode_text}</span>
                <span style="font-size: 0.78rem; color: #5e6e82; margin-top: 0.2rem; display:block;">Why skipped: {gate_msg or 'Network pressure within deterministic thresholds — Gemini call would be wasteful.'}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def _execution_summary(latest_result):
    action = latest_result.get("action", {})
    transferred = int(latest_result.get("transition", {}).get("transferred", action.get("vehicles", 0)) or 0)
    if action.get("action") == "redirect":
        return f"Executed: {transferred} vehicles redirected"
    return "Held: no transfer needed"

def _format_decision_impact(baseline_comparison):
    if not baseline_comparison:
        return "Impact will appear after the next simulation step."
    transferred = int(baseline_comparison.get("transferred", 0) or 0)
    search_delta = float(baseline_comparison.get("search_time_delta_min", 0.0) or 0.0)
    hotspot_delta = float(baseline_comparison.get("hotspot_delta", 0.0) or 0.0)
    if transferred and abs(search_delta) < 0.01:
        if hotspot_delta > 0:
            return f"Redirected {transferred} vehicle(s), prevented congestion escalation, and reduced pressure by {hotspot_delta:.0f} SRM block(s) while keeping search time stable."
        return f"Redirected {transferred} vehicle(s) and kept search time stable under load versus the no-redirect baseline."
    if transferred and search_delta > 0:
        return f"Redirected {transferred} vehicle(s) and reduced search time by {search_delta:.2f} min versus the no-redirect baseline."
    if transferred:
        return f"Redirected {transferred} vehicle(s) to protect network balance; search time changed by {search_delta:+.2f} min versus baseline."
    return "No redirect was needed; the system maintained stable local routing."

def _render_decision_card(latest_result, baseline_comparison):
    if not latest_result:
        st.info("Run one simulation step to generate the current decision.")
        return
    action = latest_result.get("action", {})
    critic = latest_result.get("critic_output", {})
    route = f"{action.get('from', '-') } -> {action.get('to', '-')}" if action.get("action") == "redirect" else "No SRM block change"
    impact = _format_decision_impact(baseline_comparison)
    # Determine Decision Label
    decision_label = "Deterministic Baseline"
    if latest_result.get("llm_advisory_used"):
        decision_label = "🧠 LLM Optimized"
    elif latest_result.get("planner_output", {}).get("memory_avoidance_triggered"):
        decision_label = "💾 Memory Filtered"
    elif not latest_result.get("execution_output", {}).get("applied") and latest_result.get("critic_output", {}).get("risk_score", 0) > 0:
        decision_label = "🛡️ Critic Vetoed"
    elif latest_result.get("planner_output", {}).get("learning_applied"):
        decision_label = "📈 Reinforcement Adaptive"

    render_key_value_groups([
        {
            "title": "Current Decision",
            "items": [
                {"label": "Insight", "value": decision_label},
                {"label": "Action", "value": action.get("action", "none").upper()},
                {"label": "Route", "value": route},
                {"label": "Vehicles", "value": action.get("vehicles", latest_result.get("transition", {}).get("transferred", 0))},
                {"label": "Reason", "value": action.get("reason", latest_result.get("reasoning", "Stable network pressure."))},
            ],
        },
        {
            "title": "Decision Impact",
            "items": [
                {"label": "Impact", "value": impact},
                {"label": "Search time delta", "value": f"{baseline_comparison.get('search_time_delta_min', 0):+.2f} min"},
                {"label": "Hotspot delta", "value": f"{baseline_comparison.get('hotspot_delta', 0):+.0f} block(s)"},
                {"label": "Execution", "value": _execution_summary(latest_result)},
            ],
        },
        {
            "title": "Safety Review",
            "items": [
                {"label": "Approved", "value": "Yes" if critic.get("approved") else "No"},
                {"label": "Risk", "value": critic.get("risk_level", "low").title()},
                {"label": "Risk score", "value": critic.get("risk_score", 0)},
                {"label": "Outcome", "value": _execution_summary(latest_result)},
            ],
        },
        {
            "title": "Learning & Adaptation",
            "items": [
                {"label": "Active Adaptation", "value": "Applied" if action.get("learning_applied") else "Passive"},
                {"label": "Learning Note", "value": latest_result.get("reward", {}).get("adaptation_note", "Learning baseline unchanged this step.")},
                {"label": "Memory Status", "value": "SRM route avoided" if latest_result.get("planner_output", {}).get("memory_avoidance_triggered") else "No route block active"},
                {"label": "Avoided Route", "value": latest_result.get("planner_output", {}).get("avoided_route", "-")},
            ],
        },
    ])
    st.success(impact)

def _render_agent_decision_table(latest_result):
    planner = latest_result.get("planner_output", {})
    critic = latest_result.get("critic_output", {})
    action = latest_result.get("action", {})
    state = latest_result.get("state", {})
    planner_action = planner.get("proposed_action") if isinstance(planner.get("proposed_action"), dict) else action
    
    # Calculate Perception Pipeline
    if state and isinstance(state, dict):
        crowded = min(state, key=lambda zone: state[zone].get("free_slots", 999))
        total_slots = state[crowded].get("total_slots", 1)
        occupancy = round((state[crowded].get("occupied", 0) / max(1, total_slots)) * 100)
        perception_string = f"{crowded} SRM block congestion {occupancy}%"
    else:
        perception_string = "SRM parking observation loaded"
        
    planner_route = f"Redirect {planner_action.get('vehicles', 0)} vehicles to {planner_action.get('to', '-')}" if planner_action.get("action") == "redirect" else "Maintain system baseline"
    critic_notes = critic.get("critic_notes", [])
    critic_reason = critic_notes[0] if critic_notes else f"Valid ({critic.get('risk_level', 'low')} risk)"
    critic_string = critic_reason if critic.get("approved") else f"Rejected: {critic_reason}"
    if planner.get("llm_requested"):
        if planner.get("llm_fallback_used") or planner.get("llm_source") == "gemini_failed_fallback":
            llm_string = "Gemini attempted -> local fallback used"
        elif planner.get("llm_advisory_used"):
            llm_string = "Gemini advisory applied"
        else:
            llm_string = "Gemini requested -> no plan change"
    else:
        llm_string = "Skipped -> deterministic agents used"
    policy_string = "Reference baseline only; used for safety recovery and benchmarking, not final authority"
    
    if action.get("status") == "failed":
        action_string = f"Execution failed: {action.get('failure_reason', 'Network timeout')}"
        action_color = "#ff9d91"
    else:
        action_string = f"Redirect {action.get('vehicles', 0)} vehicles" if action.get("action") == "redirect" else "No redirect executed"
        action_color = "#4bd38a"
    if not critic.get("approved") and action.get("action") == "redirect":
        action_string = "Blocked: critic rejection cannot execute"
        action_color = "#ff9d91"

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; font-family: monospace; line-height: 1.8;">
        <div style="color: #4da3ff;"><b>Step 1: Perception</b> &rarr; <span style="color: #ccc;">{perception_string}</span></div>
        <div style="color: #4da3ff;"><b>Step 2: LLM Gate</b> &rarr; <span style="color: #ccc;">{llm_string}</span></div>
        <div style="color: #4da3ff;"><b>Step 3: Planner</b> &rarr; <span style="color: #ccc;">Suggest {planner_route.lower()}</span></div>
        <div style="color: #4da3ff;"><b>Step 4: Critic</b> &rarr; <span style="color: #ccc;">{critic_string}</span></div>
        <div style="color: #73839a;"><b>Step 5: Baseline Context</b> &rarr; <span style="color: #ccc;">{policy_string}. If critic rejects, executor receives NONE.</span></div>
        <div style="color: {action_color};"><b>Step 6: Action</b> &rarr; <span style="color: #fff;">{action_string}</span></div>
    </div>
    """, unsafe_allow_html=True)

def _build_zone_status_frame(state_frame):
    rows = []
    for _, row in state_frame.iterrows():
        utilisation = float(row["Utilisation %"])
        if utilisation >= 90:
            status = "🔴 High pressure"
        elif utilisation >= 75:
            status = "🟡 Moderate"
        else:
            status = "🟢 Safe"
        rows.append({
            "SRM Block": row["Zone"],
            "Status": status,
            "Car Slots": int(row.get("Car Slots", 0)),
            "Bike Slots": int(row.get("Bike Slots", 0)),
            "Total Capacity": int(row["Capacity"]),
            "Free Slots": int(row["Free"]),
            "Utilisation": f"{utilisation:.1f}%",
        })
    return pd.DataFrame(rows)

def _build_srm_capacity_dataset(state_frame):
    if state_frame.empty:
        return pd.DataFrame(columns=["block_name", "car_slots", "bike_slots", "total_capacity", "occupied", "free_slots"])
    frame = state_frame.copy()
    return frame.rename(
        columns={
            "Zone": "block_name",
            "Car Slots": "car_slots",
            "Bike Slots": "bike_slots",
            "Capacity": "total_capacity",
            "Occupied": "occupied",
            "Free": "free_slots",
        }
    )[["block_name", "car_slots", "bike_slots", "total_capacity", "occupied", "free_slots"]]

def _format_live_timestamp(value):
    if not value:
        return "Awaiting live state"
    text = str(value).replace("Z", "+00:00")
    try:
        ts = pd.Timestamp(text)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata")
        return ts.strftime("%d %b %Y, %I:%M:%S %p")
    except Exception:
        return str(value)

def _render_live_slot_board(blocks, vehicles, updated_at):
    if not blocks:
        st.info("Live slot view will appear once the backend publishes a shared parking state.")
        return
    block_names = list(blocks.keys())
    selected = st.session_state.get("dashboard_block_focus")
    if selected not in blocks:
        selected = block_names[0]
    st.session_state.dashboard_block_focus = selected

    render_html_block(
        f"""
        <div class="slot-board">
            <div class="slot-selector-card">
                <div class="section-kicker">Interactive Block Selector</div>
                <div style="color:#dce8f8; font-size:1.05rem; font-weight:700;">Choose an SRM block to inspect live slots</div>
                <div class="slot-timestamp">Live backend timestamp: { _format_live_timestamp(updated_at) }</div>
            </div>
            <div class="slot-detail-card">
                <div class="section-kicker">Slot-Level Environment View</div>
                <div style="color:#dce8f8; font-size:1.05rem; font-weight:700;">Separate car and bike slots are rendered from the shared backend vehicle state.</div>
                <div class="slot-legend">
                    <span><i class="slot-swatch car"></i>Car slot</span>
                    <span><i class="slot-swatch bike"></i>Bike slot</span>
                    <span><i class="slot-swatch free"></i>Free slot</span>
                </div>
            </div>
        </div>
        """
    )

    selector_cols = st.columns(4)
    for index, block_name in enumerate(block_names):
        block = blocks.get(block_name, {})
        occupied = int(block.get("occupied", 0) or 0)
        capacity = int(block.get("capacity", 0) or 0)
        ratio = (occupied / capacity) if capacity else 0.0
        icon = "🔴" if ratio >= 0.7 else ("🟡" if ratio >= 0.4 else "🟢")
        if selector_cols[index % 4].button(
            f"{icon} {block_name}\n{occupied}/{capacity}",
            key=f"dashboard-block-{block_name}",
            use_container_width=True,
            on_click=_mark_ui_interaction,
        ):
            st.session_state.dashboard_block_focus = block_name
            selected = block_name

    selected = st.session_state.get("dashboard_block_focus", selected)
    block = blocks.get(selected, {})
    block_capacity = int(block.get("capacity", 0) or 0)
    car_slots = min(block_capacity, int(block.get("car_slots", block_capacity) or block_capacity))
    bike_slots = max(0, int(block.get("bike_slots", max(0, block_capacity - car_slots)) or 0))
    block_vehicles = [vehicle for vehicle in vehicles if vehicle.get("block") == selected]
    slot_lookup = {int(vehicle.get("slot", 0) or 0): vehicle for vehicle in block_vehicles}
    slot_html = []
    for slot_number in range(1, block_capacity + 1):
        vehicle = slot_lookup.get(slot_number)
        slot_type = "bike" if slot_number > car_slots else "car"
        if vehicle:
            klass = "filled-bike" if vehicle.get("type") == "bike" else "filled-car"
            icon = "🏍" if vehicle.get("type") == "bike" else "🚗"
        else:
            klass = "empty"
            icon = "·"
        slot_html.append(
            f"<div class='slot-cell {klass}'><div>{icon}</div><small>{slot_number}</small></div>"
        )

    cars_live = sum(1 for vehicle in block_vehicles if vehicle.get("type") == "car")
    bikes_live = sum(1 for vehicle in block_vehicles if vehicle.get("type") == "bike")
    render_html_block(
        f"""
        <div class="slot-detail-card">
            <div class="focus-kicker">Selected SRM Block</div>
            <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <div class="focus-title" style="font-size:2rem; margin-bottom:0.3rem;">{selected}</div>
                    <div class="focus-route">Occupied {int(block.get("occupied", 0) or 0)} of {block_capacity} | Free {int(block.get("free_slots", 0) or 0)}</div>
                </div>
                <div class="focus-stat-grid" style="margin-top:0; min-width:320px;">
                    <div class="focus-stat"><span>Car Slots</span><strong>{car_slots}</strong></div>
                    <div class="focus-stat"><span>Bike Slots</span><strong>{bike_slots}</strong></div>
                    <div class="focus-stat"><span>Live Vehicles</span><strong>{len(block_vehicles)}</strong></div>
                    <div class="focus-stat"><span>Cars Parked</span><strong>{cars_live}</strong></div>
                    <div class="focus-stat"><span>Bikes Parked</span><strong>{bikes_live}</strong></div>
                    <div class="focus-stat"><span>Entry / Exit</span><strong>{int(block.get("entry", 0) or 0)} / {int(block.get("exit", 0) or 0)}</strong></div>
                </div>
            </div>
            <div class="slot-grid-dash">{''.join(slot_html)}</div>
        </div>
        """
    )

def _summarize_agent_rows(latest_result, agent_loop_steps):
    rows = _safe_list(latest_result.get("agent_interactions"))
    if not rows and agent_loop_steps:
        rows = [
            {
                "Agent": item.get("step", "Agent"),
                "Action Taken": item.get("output", "Pending"),
                "Why": item.get("details", {}),
                "Key Output": item.get("output", ""),
            }
            for item in agent_loop_steps
        ]
    ordered = []
    seen = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        agent = item.get("Agent") or item.get("agent") or "Agent"
        normalized = {
            "agent": str(agent),
            "action": str(item.get("Action Taken", item.get("message", "Pending"))),
            "why": _format_reasoning_text(item.get("Why", item.get("why", ""))),
            "signal": str(item.get("Key Output", "")),
        }
        if agent not in seen:
            ordered.append(agent)
        seen[agent] = normalized
    return [seen[name] for name in ordered]

def _build_movement_frame(movement_log):
    rows = []
    for item in movement_log or []:
        if not isinstance(item, dict):
            continue
        timestamp = _format_live_timestamp(item.get("timestamp"))
        rows.append(
            {
                "Timestamp": timestamp,
                "Step": item.get("step", "-"),
                "SRM Block": item.get("block", "-"),
                "Entries": int(item.get("entries", 0) or 0),
                "Exits": int(item.get("exits", 0) or 0),
                "Car In": int(item.get("car_entries", 0) or 0),
                "Bike In": int(item.get("bike_entries", 0) or 0),
                "Car Out": int(item.get("car_exits", 0) or 0),
                "Bike Out": int(item.get("bike_exits", 0) or 0),
                "Occupied After": int(item.get("occupied_after", 0) or 0),
            }
        )
    return pd.DataFrame(rows)

def _render_decision_audit(latest_result):
    if not latest_result:
        st.info("Run one simulation step to generate the decision audit.")
        return

    action = latest_result.get("action", {})
    planner = latest_result.get("planner_output", {})
    critic = latest_result.get("critic_output", {})
    executor = latest_result.get("execution_output", {})
    budget = latest_result.get("reasoning_budget", {})
    provenance = latest_result.get("decision_provenance", {})
    baseline = latest_result.get("baseline_comparison", {})
    reward_impact = latest_result.get("reward_impact", {})
    model_alignment = latest_result.get("model_alignment", {})

    route = f"{action.get('from', '-')} -> {action.get('to', '-')}" if action.get("action") == "redirect" else "-"
    render_key_value_groups([
        {
            "title": "Final Decision",
            "items": [
                {"label": "Action", "value": action.get("action", "none").upper()},
                {"label": "Route", "value": route},
                {"label": "Vehicles", "value": action.get("vehicles", 0)},
                {"label": "Reasoning mode", "value": budget.get("budget_level", "local_only")},
            ],
        },
        {
            "title": "LLM Use",
            "items": [
                {"label": "Planner Gemini requested", "value": "Yes" if planner.get("llm_requested") else "No"},
                {"label": "Planner Gemini changed plan", "value": "Yes" if planner.get("llm_advisory_used") else "No"},
                {"label": "Planner fallback used", "value": "Yes" if planner.get("llm_fallback_used") else "No"},
                {"label": "Critic Gemini requested", "value": "Yes" if critic.get("llm_requested") else "No"},
                {"label": "Critic Gemini changed review", "value": "Yes" if critic.get("llm_advisory_used") else "No"},
                {"label": "Fallback reason", "value": planner.get("llm_fallback_reason") or critic.get("llm_fallback_reason") or "-"},
            ],
        },
        {
            "title": "Decision Provenance",
            "items": [
                {"label": "Origin", "value": provenance.get("decision_origin", "-").replace("_", " ").title()},
                {"label": "Final authority", "value": provenance.get("final_authority", "-").title()},
                {"label": "Memory influenced", "value": "Yes" if provenance.get("memory_influenced") else "No"},
                {"label": "Critic changed action", "value": "Yes" if provenance.get("critic_changed_action") else "No"},
                {"label": "Controller override", "value": "Yes" if provenance.get("controller_override") else "No"},
                {"label": "Fallback used", "value": "Yes" if provenance.get("fallback_used") else "No"},
            ],
        },
        {
            "title": "Execution Result",
            "items": [
                {"label": "Approved", "value": "Yes" if critic.get("approved") else "No"},
                {"label": "Risk", "value": critic.get("risk_level", "low").upper()},
                {"label": "Execution", "value": _execution_summary(latest_result)},
                {"label": "Reward", "value": latest_result.get("reward_score", 0)},
            ],
        },
    ])

    if baseline:
        before_after = [
            {
                "title": "Before vs After",
                "items": [
                    {"label": "No-redirect search time", "value": f"{baseline.get('baseline_kpis', {}).get('estimated_search_time_min', 0)} min"},
                    {"label": "Agent search time", "value": f"{baseline.get('agent_kpis', {}).get('estimated_search_time_min', 0)} min"},
                    {"label": "Search time delta", "value": f"{baseline.get('search_time_delta_min', 0):+.2f} min"},
                    {"label": "Resilience delta", "value": f"{baseline.get('resilience_delta', 0):+.2f}"},
                ],
            }
        ]
        render_key_value_groups(before_after)

    st.write(action.get("reason") or latest_result.get("reasoning", "No explanation available."))
    
    if action.get("action") == "redirect":
        st.markdown("**Why not other options?**")
        st.info("Alternative routes were available but not selected due to lower pressure alignment or queue risk at the destination.")
        
    if model_alignment:
        st.markdown("**Why agents may disagree**")
        st.info(model_alignment.get("explanation", "No model alignment explanation available."))
    if planner.get("rationale"):
        st.markdown("**Planner rationale**")
        st.write(planner.get("rationale"))
    critic_notes = critic.get("critic_notes", [])
    if critic_notes:
        st.markdown("**Critic notes**")
        for note in critic_notes:
            st.markdown(f"- {note}")
    st.markdown("**Execution**")
    st.write(_execution_summary(latest_result))
    if reward_impact:
        st.markdown("**Reward impact**")
        reward_text = f"{reward_impact.get('direction', 'neutral').title()} reward: {reward_impact.get('score', 0):+.3f}. {reward_impact.get('explanation', '')}"
        if reward_impact.get("direction") == "positive":
            st.success(reward_text)
        elif reward_impact.get("direction") == "negative":
            st.warning(reward_text)
        else:
            st.info(reward_text)
    llm_context = _safe_dict(latest_result.get("llm", {}))
    influence_summary = llm_context.get("influence_summary")
    if influence_summary:
        st.markdown("**LLM influence**")
        st.info(influence_summary)

def _render_decision_explainability(explanation):
    if not explanation:
        st.info("Run one simulation step to generate a structured decision explanation.")
        return
    signals = _safe_dict(explanation.get("current_signals"))
    impact = _safe_dict(explanation.get("expected_impact"))
    safety = _safe_dict(explanation.get("safety_review"))
    llm_context = _safe_dict(explanation.get("llm_context"))
    render_key_value_groups([
        {
            "title": "Why This Decision",
            "items": [
                {"label": "Decision", "value": explanation.get("headline", "-")},
                {"label": "Reason", "value": explanation.get("why_this_decision", "-")},
                {"label": "Scenario", "value": signals.get("scenario", "-")},
                {"label": "Reasoning budget", "value": signals.get("reasoning_budget", "local_only")},
            ],
        },
        {
            "title": "Signal Check",
            "items": [
                {"label": "Search time", "value": f"{signals.get('search_time_min', 0)} min"},
                {"label": "Hotspots", "value": signals.get("congestion_hotspots", 0)},
                {"label": "Queue", "value": signals.get("queue_length", 0)},
                {"label": "Risk", "value": safety.get("risk_level", "low")},
            ],
        },
        {
            "title": "Expected Impact",
            "items": [
                {"label": "Search delta", "value": f"{impact.get('search_time_delta_min', 0):+.2f} min"},
                {"label": "Hotspot delta", "value": f"{impact.get('hotspot_delta', 0):+.0f}"},
                {"label": "Resilience delta", "value": f"{impact.get('resilience_delta', 0):+.2f}"},
                {"label": "Reward", "value": f"{impact.get('reward_score', 0):+.2f}"},
            ],
        },
        {
            "title": "LLM Context",
            "items": [
                {"label": "Source", "value": llm_context.get("source", "deterministic")},
                {"label": "Requested", "value": "Yes" if llm_context.get("planner_requested") else "No"},
                {"label": "Used", "value": "Yes" if llm_context.get("planner_used") else "No"},
                {"label": "Fallback", "value": "Yes" if llm_context.get("fallback_used") else "No"},
                {"label": "Influence", "value": llm_context.get("influence_summary") or llm_context.get("summary", "-")},
            ],
        },
    ])
    st.markdown("**Agent Chain**")
    st.dataframe(pd.DataFrame(_safe_list(explanation.get("agent_chain"))), width="stretch", hide_index=True)
    alternatives = _safe_list(explanation.get("alternatives_considered"))
    if alternatives:
        st.markdown("**Alternatives Considered**")
        for item in alternatives:
            st.markdown(f"- {item}")
    st.info(explanation.get("why_not_other_options", "No alternative-route explanation available."))

def _group_notification_dispatch(dispatch_rows):
    grouped = {}
    for row in dispatch_rows:
        key = (row.get("event"), row.get("title"), row.get("message"), row.get("level"))
        grouped.setdefault(key, {"channels": [], "latest_timestamp": row.get("timestamp", "")})
        grouped[key]["channels"].append(row.get("channel"))
        grouped[key]["latest_timestamp"] = max(grouped[key]["latest_timestamp"], row.get("timestamp", ""))
    records = []
    for (event, title, message, level), data in grouped.items():
        records.append({
            "Event": event, "Alert": title,
            "Channels": ", ".join(sorted(set(data["channels"]))),
            "Level": level,
            "Latest": data["latest_timestamp"],
            "Message": message,
        })
    return pd.DataFrame(records)

def _chart_key(name, step_number, suffix=""):
    suffix_part = f"-{suffix}" if suffix else ""
    return f"{name}-step-{step_number}{suffix_part}"

def _safe_dict(value):
    return value if isinstance(value, dict) else {}

def _safe_list(value):
    return value if isinstance(value, list) else []

def _fallback_snapshot():
    from environment.parking_environment import ParkingEnvironment

    env = ParkingEnvironment(seed=0)
    transition = env.get_last_transition()
    return {
        "state": env.get_state(),
        "latest_result": {
            "state": env.get_state(),
            "transition": transition,
            "planner_output": {},
            "critic_output": {},
            "execution_output": {},
            "agent_interactions": [],
            "reasoning_budget": {},
        },
        "latest_transition": transition,
        "metrics": {"steps": 0},
        "goal": {},
        "scenario_mode": env.get_scenario_mode(),
        "llm_status": {"available": False, "message": "Local fallback snapshot active."},
        "llm_mode": "auto",
        "force_llm": False,
        "event_context": env.get_event_context(),
        "notifications": [],
        "alerts": [],
        "notification_dispatch": [],
        "kpis": {},
        "recent_cycles": [],
        "recent_states": [env.get_state()],
        "trace": [],
        "benchmark": {},
        "benchmark_summary": {"message": "Run the benchmark from the sidebar to generate metrics."},
        "reasoning_summary": {},
        "agent_loop_steps": [],
        "memory_summary": {},
        "notification_summary": {},
        "last_llm_decision": {},
        "llm_usage_summary": {},
        "assistant_briefing": {},
        "decision_explanation": {},
        "vehicles": [],
        "movement_log": [],
        "actions": [],
        "updated_at": transition.get("timestamp", ""),
    }

def _normalize_dashboard_snapshot(snapshot):
    if not isinstance(snapshot, dict) or not snapshot:
        return _fallback_snapshot()
    normalized = dict(snapshot)
    fallback = _fallback_snapshot()
    for key, value in fallback.items():
        normalized.setdefault(key, value)
    normalized["state"] = _safe_dict(normalized.get("state")) or fallback["state"]
    normalized["latest_result"] = _safe_dict(normalized.get("latest_result"))
    normalized["latest_transition"] = _safe_dict(normalized.get("latest_transition"))
    normalized["latest_result"]["state"] = normalized["state"]
    normalized["latest_result"].setdefault("transition", normalized["latest_transition"])
    for key in ["planner_output", "critic_output", "execution_output", "reasoning_budget"]:
        normalized["latest_result"].setdefault(key, {})
    normalized["latest_result"].setdefault("agent_interactions", [])
    normalized["recent_states"] = _safe_list(normalized.get("recent_states")) or [normalized["state"]]
    normalized["recent_cycles"] = _safe_list(normalized.get("recent_cycles"))
    normalized["trace"] = _safe_list(normalized.get("trace"))
    normalized.setdefault("blocks", normalized["state"])
    normalized.setdefault("vehicles", [])
    normalized.setdefault("movement_log", [])
    normalized.setdefault("actions", [])
    normalized.setdefault("alerts", [])
    normalized.setdefault("updated_at", "")
    normalized.setdefault("decision_explanation", {})
    return normalized

def _safe_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return value
    except TypeError:
        return value if value is not None else default

def _format_reasoning_text(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("rationale", "reason", "message", "summary"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        compact = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            compact.append(f"{key}: {item}")
            if len(compact) >= 3:
                break
        if compact:
            return " | ".join(compact)
    return "Waiting for reasoning..."

def _normalize_recent_states(recent_states, current_state, current_kpis, step_number):
    normalized = []
    for index, item in enumerate(recent_states or []):
        if not isinstance(item, dict):
            continue
        kpis = _safe_dict(item.get("kpis"))
        if not kpis and item == current_state:
            kpis = current_kpis
        normalized.append({
            **item,
            "step": item.get("step", index if step_number == 0 else step_number - len(recent_states) + index + 1),
            "kpis": kpis,
        })
    if not normalized:
        normalized = [{"step": step_number, "state": current_state, "kpis": current_kpis}]
    elif normalized[-1].get("state") != current_state and normalized[-1] != current_state:
        normalized.append({"step": step_number, "state": current_state, "kpis": current_kpis})
    return normalized

def _build_dashboard_view_model(snapshot, state_frame, step_number):
    latest_result = _safe_dict(snapshot.get("latest_result"))
    latest_transition = _safe_dict(snapshot.get("latest_transition"))
    current_state = _safe_dict(snapshot.get("state"))
    kpis = _safe_dict(snapshot.get("kpis"))
    agent_trace = _safe_list(snapshot.get("agent_loop_steps"))
    if not agent_trace:
        agent_trace = _safe_list(latest_result.get("agent_interactions"))
    memory = _safe_dict(snapshot.get("memory_summary"))
    learning = _safe_dict(memory.get("learning_profile"))
    recent_states = _normalize_recent_states(
        _safe_list(snapshot.get("recent_states")),
        current_state,
        kpis,
        step_number,
    )
    return {
        "blocks": current_state,
        "state_frame": state_frame,
        "capacity": int(state_frame["Capacity"].sum()) if not state_frame.empty else 0,
        "occupied": int(state_frame["Occupied"].sum()) if not state_frame.empty else 0,
        "free_slots": int(state_frame["Free"].sum()) if not state_frame.empty else 0,
        "metrics": kpis,
        "runtime_metrics": _safe_dict(snapshot.get("metrics")),
        "latest_result": latest_result,
        "latest_transition": latest_transition,
        "agent_trace": agent_trace,
        "planner_output": _safe_dict(latest_result.get("planner_output")),
        "critic_output": _safe_dict(latest_result.get("critic_output")),
        "memory": memory,
        "learning": learning,
        "blocked_routes": _safe_list(learning.get("blocked_routes")),
        "recent_states": recent_states,
        "recent_cycles": _safe_list(snapshot.get("recent_cycles")),
        "benchmark": _safe_dict(snapshot.get("benchmark")),
        "benchmark_summary": _safe_dict(snapshot.get("benchmark_summary")),
        "event_context": _safe_dict(snapshot.get("event_context")),
        "goal": _safe_dict(snapshot.get("goal")),
        "notifications": _safe_list(snapshot.get("notifications")),
        "alerts": _safe_list(snapshot.get("alerts")),
        "vehicles": _safe_list(snapshot.get("vehicles")),
        "movement_log": _safe_list(snapshot.get("movement_log")),
        "actions": _safe_list(snapshot.get("actions")),
        "updated_at": snapshot.get("updated_at", ""),
        "decision_explanation": _safe_dict(snapshot.get("decision_explanation")),
    }

def main():
    st.set_page_config(layout="wide", page_title=STRINGS["title"])
    inject_styles()
    _ensure_session_state()

    st.markdown(
        f"""
        <div class="hero-badge">Agent Status: ACTIVE DECISION MODE</div>
        <div class="hero-title">{STRINGS["title"]}</div>
        <div class="hero-sub">{STRINGS["subtitle"]}</div>
        """,
        unsafe_allow_html=True,
    )

    snapshot = api_bridge.get_snapshot()

    if not snapshot:
        st.warning("Simulation core returned no snapshot. Rendering the SRM block fallback state instead of a blank dashboard.")
    snapshot = _normalize_dashboard_snapshot(snapshot)

    current_scenario = snapshot.get("scenario_mode", "Auto Schedule")
    metrics = snapshot.get("metrics", {})
    step_number = metrics.get("steps", 0)

    st.sidebar.header("Simulation Controls")
    selected_scenario = st.sidebar.selectbox(
        "SRM Parking Scenario",
        SCENARIOS,
        index=SCENARIOS.index(current_scenario) if current_scenario in SCENARIOS else 0,
        key="sidebar_scenario",
        on_change=_mark_ui_interaction,
    )
    if selected_scenario != current_scenario:
        if api_bridge.set_scenario(selected_scenario):
            st.rerun()

    current_llm_mode = snapshot.get("llm_mode", "auto")
    llm_labels = {"auto": "Auto", "demo": "Demo Mode", "local": "Local Only"}
    label_to_mode = {value: key for key, value in llm_labels.items()}
    selected_llm_label = st.sidebar.radio(
        "LLM Mode",
        list(label_to_mode.keys()),
        index=list(label_to_mode.keys()).index(llm_labels.get(current_llm_mode, "Auto")),
        horizontal=True,
        key="sidebar_llm_mode",
        on_change=_mark_ui_interaction,
    )
    selected_llm_mode = label_to_mode[selected_llm_label]
    if selected_llm_mode != current_llm_mode:
        if api_bridge.set_llm_mode(selected_llm_mode):
            st.rerun()

    force_llm_active = snapshot.get("force_llm", False)
    new_force_llm = st.sidebar.checkbox(
        "⚡ Strategic Overdrive (Force LLM)",
        value=force_llm_active,
        help="Bypasses app-level efficiency logic and forces a live Gemini attempt on every single step. Provider quota limits can still return 429.",
        key="sidebar_force_llm",
        on_change=_mark_ui_interaction,
    )
    if new_force_llm != force_llm_active:
        if api_bridge.set_force_llm(new_force_llm):
            st.rerun()

    if st.sidebar.button("🔄 Reset AI Quota Cooldown", help="Clear local 429 backoffs and retry Gemini immediately.", on_click=_mark_ui_interaction):
        api_bridge.reset_llm_runtime_state()
        st.toast("AI runtime state reset. Attempting fresh Gemini link...")
        st.rerun()

    llm_stride = snapshot.get("latest_result", {}).get("reasoning_budget", {}).get("signals", {}).get("llm_stride_steps", 10)
    st.sidebar.caption(
        f"Budget-Aware Adaptive Invocation: Gemini recalibrates every {llm_stride} steps and only escalates early for queue >= 3, entropy > 3.5, risk > 70, or decision conflict."
    )

    st.session_state.run = st.sidebar.toggle("Autonomous Mode", value=st.session_state.run, key="sidebar_autonomous_mode", on_change=_mark_ui_interaction)
    speed = st.sidebar.slider("Step Interval (seconds)", 3.0, 12.0, 6.0, key="sidebar_step_interval", on_change=_mark_ui_interaction)
    st.sidebar.caption("Use Run One Step first. For a polished project demo, 5-7 seconds looks smoother and easier to follow.")
    
    benchmark_episodes = st.sidebar.slider("Benchmark Episodes", 1, 5, 3, key="sidebar_benchmark_episodes", on_change=_mark_ui_interaction)
    benchmark_steps = st.sidebar.slider("Benchmark Steps", 6, 15, 10, key="sidebar_benchmark_steps", on_change=_mark_ui_interaction)
    
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Run One Step", width="stretch", on_click=_mark_ui_interaction):
        with st.spinner("Tick..."):
            api_bridge.step_simulation()
        st.rerun()
        
    if c2.button("Pause", width="stretch", on_click=_mark_ui_interaction):
        st.session_state.run = False
        st.rerun()

    c3, c4 = st.sidebar.columns(2)
    if c3.button("Resume", width="stretch", on_click=_mark_ui_interaction):
        st.session_state.run = True
        st.session_state.last_run = 0.0
        st.rerun()
    if c4.button("Reset Runtime", width="stretch", on_click=_mark_ui_interaction):
        api_bridge.reset(clear_memory=False)
        st.session_state.run = False
        st.rerun()

    if st.sidebar.button("Run Benchmark", width="stretch", on_click=_mark_ui_interaction):
        with st.spinner("Benchmarking Agents vs Baseline..."):
            api_bridge.run_benchmark(benchmark_episodes, benchmark_steps)
            st.session_state.benchmark_toggle = not st.session_state.benchmark_toggle
        st.session_state.run = False
        st.rerun()

    if st.sidebar.button("Prepare Run Report", width="stretch", on_click=_mark_ui_interaction):
        report = api_bridge.get_run_report()
        st.session_state.run_report_json = json.dumps(report, indent=2) if report else ""
    if st.session_state.run_report_json:
        st.sidebar.download_button(
            "Download Run Report JSON",
            data=st.session_state.run_report_json,
            file_name="srm_agentic_run_report.json",
            mime="application/json",
            width="stretch",
        )

    st.sidebar.divider()
    st.session_state.developer_mode = st.sidebar.toggle("🖥️ Developer Mode", value=st.session_state.get('developer_mode', False), key="sidebar_developer_mode", on_change=_mark_ui_interaction)

    now = time.time()
    if st.session_state.run and not _ui_recently_interacted(now) and now - st.session_state.last_run >= speed:
        st.session_state.last_run = now
        api_bridge.step_simulation()
        snapshot = _normalize_dashboard_snapshot(api_bridge.get_snapshot()) # Refresh locally during tick instead of full rerun cache miss
        metrics = snapshot.get("metrics", {})
        step_number = metrics.get("steps", 0)

    state = _safe_dict(snapshot.get("state"))
    latest_result = _safe_dict(snapshot.get("latest_result"))
    latest_transition = _safe_dict(snapshot.get("latest_transition"))
    goal = _safe_dict(snapshot.get("goal"))
    event_context = _safe_dict(snapshot.get("event_context"))
    kpis = _safe_dict(snapshot.get("kpis"))
    notifications = _safe_list(snapshot.get("notifications"))
    recent_cycles = _safe_list(snapshot.get("recent_cycles"))
    recent_states = _safe_list(snapshot.get("recent_states")) or [state]
    llm_status = _safe_dict(snapshot.get("llm_status"))
    llm_mode = snapshot.get("llm_mode", "auto")
    benchmark = _safe_dict(snapshot.get("benchmark"))
    benchmark_summary = _safe_dict(snapshot.get("benchmark_summary"))
    assistant_briefing = _safe_dict(snapshot.get("assistant_briefing"))
    operational_signals = _safe_dict(latest_result.get("operational_signals", latest_transition.get("dynamic_signals", {})))
    baseline_comparison = _safe_dict(latest_result.get("baseline_comparison"))
    reasoning_summary = _safe_dict(snapshot.get("reasoning_summary"))
    agent_loop_steps = _safe_list(snapshot.get("agent_loop_steps"))
    memory_summary = _safe_dict(snapshot.get("memory_summary"))
    notification_summary = _safe_dict(snapshot.get("notification_summary"))
    last_llm_decision = _safe_dict(snapshot.get("last_llm_decision"))
    llm_usage_summary = _safe_dict(snapshot.get("llm_usage_summary"))
    decision_explanation = _safe_dict(snapshot.get("decision_explanation"))

    force_llm = snapshot.get("force_llm", False)
    if force_llm:
        st.sidebar.success("⚡ Strategic Overdrive: Active")
        st.sidebar.caption("Live Gemini attempts are forced on every step. App cooldowns are ignored while this mode is active.")
    elif llm_status.get("quota_backoff", {}).get("active"):
        backoff = llm_status.get("quota_backoff", {})
        st.sidebar.warning(
            f"Gemini cooldown: {backoff.get('kind', 'backoff')} for {backoff.get('remaining_seconds', 0)} sec."
        )
        if llm_status.get("last_error"):
            st.sidebar.caption(f"Last LLM error: {llm_status.get('last_error')}")
    elif llm_status.get("available"):
        st.sidebar.success(f"Gemini active: {llm_status.get('model', 'gemini')}")
    else:
        st.sidebar.info(f"Local mode: {llm_status.get('message', 'No status available.')}")
    if llm_mode == "auto":
        st.sidebar.info("Smart LLM Policy is active: Gemini is invoked adaptively with a mandatory heartbeat every 10 steps.")
    elif force_llm:
        st.sidebar.info("Overdrive mode overrides local gating. Every step will visibly attempt live Gemini reasoning.")

    # Cached heavy processing bound strictly to the step number cache key
    state_frame = state_manager.get_state_frame(state, step_number)
    if state_frame.empty:
        st.error("No SRM parking block state is available. Reset the runtime to reload the block dataset.")
        st.stop()

    dashboard_state = _build_dashboard_view_model(snapshot, state_frame, step_number)
    st.session_state.dashboard_state = dashboard_state
    state = dashboard_state["blocks"]
    state_frame = dashboard_state["state_frame"]
    latest_result = dashboard_state["latest_result"]
    latest_transition = dashboard_state["latest_transition"]
    goal = dashboard_state["goal"]
    event_context = dashboard_state["event_context"]
    kpis = dashboard_state["metrics"]
    notifications = dashboard_state["notifications"]
    alerts = dashboard_state["alerts"]
    recent_cycles = dashboard_state["recent_cycles"]
    recent_states = dashboard_state["recent_states"]
    benchmark = dashboard_state["benchmark"]
    benchmark_summary = dashboard_state["benchmark_summary"]
    agent_loop_steps = dashboard_state["agent_trace"]
    memory_summary = dashboard_state["memory"]
    vehicles = dashboard_state["vehicles"]
    movement_log = dashboard_state["movement_log"]
    actions = dashboard_state["actions"]
    updated_at = dashboard_state["updated_at"]
    decision_explanation = _safe_dict(dashboard_state.get("decision_explanation"))

    total_capacity = dashboard_state["capacity"]
    total_occupied = dashboard_state["occupied"]
    total_free = dashboard_state["free_slots"]
    congestion = round((total_occupied / total_capacity * 100), 2) if total_capacity else 0.0

    st.markdown(
        f"""
        <div class="event-banner">
            <div class="event-title">SRM Parking Simulation: {event_context.get("name", "SRM Simulation")} | Live at {_format_live_timestamp(updated_at)}</div>
            <div class="event-copy">
                {event_context.get("description", "")}
                Strategy: <strong>{event_context.get("allocation_strategy", "Balanced utilisation")}</strong>.
                Recommended SRM block for incoming vehicles: <strong>{event_context.get("recommended_zone", "-")}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_srm_parking_overview(state_frame)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("SRM Total Capacity", total_capacity)
    m2.metric("Occupied", total_occupied)
    m3.metric("Free Slots", total_free)
    m4.metric("SRM Occupancy %", congestion)
    m5.metric("Saved Steps", step_number)

    signal_cards(event_context, latest_result, kpis, goal)
    _render_system_status_bar(latest_result, event_context, kpis, llm_status, llm_mode)
    _render_decision_summary_block(latest_result, baseline_comparison, updated_at)

    backoff = llm_status.get("quota_backoff", {})
    if backoff.get("active") and backoff.get("kind") == "daily_quota":
        render_html_block(
            """
            <div class="quota-panel">
                <strong>Live Gemini is paused by daily quota.</strong>
                <span>The system is still running through simulated and local multi-agent reasoning, so execution remains continuous while the provider quota is exhausted.</span>
            </div>
            """
        )

    if st.session_state.run:
        if speed < 3:
            st.warning("Autonomous mode is running quickly. For free-tier Gemini, 3-5 seconds or manual steps are safer.")
        else:
            st.info(f"Autonomous mode is active. The simulation refreshes every {speed:.1f} seconds.")

    dashboard_pages = ["SRM Operations", "Events & KPIs", "Benchmark", "Agent Loop", "Reasoning", "Memory & Goals", "Vehicle Flow", "Notifications", "AI Chat"]
    current_page = st.session_state.get("active_dashboard_page", "SRM Operations")
    if current_page not in dashboard_pages:
        current_page = "SRM Operations"
    nav_left, nav_center, nav_right = st.columns([0.14, 0.72, 0.14])
    with nav_left:
        if st.button("Previous", width="stretch", on_click=_mark_ui_interaction):
            current_index = dashboard_pages.index(current_page)
            current_page = dashboard_pages[max(0, current_index - 1)]
    with nav_center:
        active_page = st.selectbox(
            "Dashboard section",
            dashboard_pages,
            index=dashboard_pages.index(current_page),
            label_visibility="collapsed",
            key="dashboard_section",
            on_change=_mark_ui_interaction,
        )
        current_page = active_page
    with nav_right:
        if st.button("Next", width="stretch", on_click=_mark_ui_interaction):
            current_index = dashboard_pages.index(current_page)
            current_page = dashboard_pages[min(len(dashboard_pages) - 1, current_index + 1)]
    active_page = current_page
    st.session_state.active_dashboard_page = active_page

    if active_page == "SRM Operations":
        st.markdown("<div class='section-kicker'>Live SRM Parking Operations</div>", unsafe_allow_html=True)
        _render_decision_card(latest_result, baseline_comparison)
        _render_llm_insight(latest_result)
        
        if baseline_comparison:
            baseline = baseline_comparison.get('baseline_kpis', {})
            agent_kpis = baseline_comparison.get('agent_kpis', {})
            before_after = [
                {
                    "title": "Before vs After Snapshot",
                    "items": [
                        {"label": "No-redirect baseline", "value": f"{baseline.get('estimated_search_time_min', 0):.1f} min"},
                        {"label": "Agent decision applied", "value": f"{agent_kpis.get('estimated_search_time_min', 0):.1f} min"},
                        {"label": "Congested SRM blocks", "value": f"{baseline.get('congestion_hotspots', 0)} → {agent_kpis.get('congestion_hotspots', 0)}"},
                    ],
                }
            ]
            render_key_value_groups(before_after)
        
        llm_title, llm_body, llm_level = _build_llm_summary(llm_status, is_forced=force_llm)
        if llm_level == "success": st.success(f"**{llm_title}**\n\n{llm_body}")
        else: st.warning(f"**{llm_title}**\n\n{llm_body}")
        
        with st.expander("Copilot prediction and next-step advice (What might happen next)", expanded=False):
            _render_assistant_briefing(assistant_briefing)

        st.markdown("<div class='section-kicker'>Agent Decisions</div>", unsafe_allow_html=True)
        _render_agent_decision_table(latest_result)
        with st.expander("Decision explainability: why this action was selected", expanded=False):
            _render_decision_explainability(decision_explanation)

        st.markdown("<div class='section-kicker'>SRM Block Pressure</div>", unsafe_allow_html=True)
        st.dataframe(_build_zone_status_frame(state_frame), width="stretch", hide_index=True)
        st.markdown("<div class='section-kicker'>Live Slot-Level Parking View</div>", unsafe_allow_html=True)
        _render_live_slot_board(state, vehicles, updated_at)

        left, right = st.columns([1, 1])
        kpi_chart = build_kpi_chart(recent_states)
        latest_chart = build_latest_baseline_chart(baseline_comparison)
        if kpi_chart is not None:
            left.plotly_chart(
                kpi_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("operations-kpi", step_number),
            )
        else:
            left.info("Run a few steps to plot search time and congestion trend.")
        if latest_chart is not None:
            right.plotly_chart(
                latest_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("operations-baseline", step_number),
            )
        else:
            right.plotly_chart(
                build_utilisation_chart(state_frame),
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("operations-utilisation", step_number),
            )

        with st.expander("Advanced View: SRM block cards, occupancy chart, and detailed table", expanded=False):
            render_story_cards(goal, latest_result, event_context, kpis)
            render_zone_cards(state_frame)
            st.plotly_chart(
                build_zone_chart(state_frame),
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("operations-zone", step_number),
            )
            st.dataframe(styled_state_frame(state_frame), width="stretch", hide_index=True)

        flow_chart = build_flow_chart(recent_states)
        if flow_chart is not None:
            with st.expander("Advanced View: raw entries, exits, and redirects"):
                st.plotly_chart(
                    flow_chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=_chart_key("operations-flow", step_number),
                )

        lower_left, lower_right = st.columns([1, 1])
        lower_left.markdown("<div class='section-kicker'>Latest SRM Block Transition</div>", unsafe_allow_html=True)
        
        transition_frame = state_manager.get_transition_frame(latest_transition.get("zones", []), step_number)
        lower_left.dataframe(styled_transition_frame(transition_frame), width="stretch", hide_index=True)
        
        with lower_right:
            st.markdown("<div class='section-kicker'>Latest Allocation Summary</div>", unsafe_allow_html=True)
            render_key_value_groups(
                [
                    {
                        "title": "Allocation Result",
                        "items": [
                            {"label": "Transferred vehicles", "value": latest_transition.get("transferred", 0)},
                            {"label": "Denied entries", "value": latest_transition.get("totals", {}).get("denied_entries", 0)},
                        ],
                    },
                    {
                        "title": "Routing Context",
                        "items": [
                            {"label": "Recommended SRM block", "value": event_context.get("recommended_zone", "-")},
                            {"label": "Focus SRM block", "value": event_context.get("focus_zone", "-")},
                        ],
                    },
                ]
            )

    elif active_page == "Events & KPIs":
        st.markdown("<div class='section-kicker'>SRM Parking Blocks & Slots</div>", unsafe_allow_html=True)
        st.dataframe(_build_srm_capacity_dataset(state_frame), width="stretch", hide_index=True)
        st.markdown("<div class='section-kicker'>Live Car and Bike Slot View</div>", unsafe_allow_html=True)
        _render_live_slot_board(state, vehicles, updated_at)

        st.markdown("<div class='section-kicker'>SRM Outcome Metrics</div>", unsafe_allow_html=True)
        render_insight_cards([
            {"title": "Search Time", "value_label": "Estimated", "value": f"{kpis.get('estimated_search_time_min', 0)} min", "note": "Lower is better."},
            {"title": "Space Utilisation", "value_label": "Current", "value": f"{kpis.get('space_utilisation_pct', 0)}%", "note": "Healthy utilisation."},
            {"title": "Allocation Success", "value_label": "Current", "value": f"{kpis.get('allocation_success_pct', 0)}%", "note": "Reallocation success."},
            {"title": "SRM Block Hotspots", "value_label": "Current", "value": str(kpis.get('congestion_hotspots', 0)), "note": "Blocks deeply congested."},
        ], columns=4)

        kpi_chart = build_kpi_chart(recent_states)
        if kpi_chart is not None:
            st.plotly_chart(
                kpi_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("events-kpi", step_number),
            )
        else:
            st.info("Metrics history is not populated yet. Run one step to draw SRM search-time and hotspot trends.")

    elif active_page == "Benchmark":
        st.markdown("<div class='section-kicker'>Simulation Proof</div>", unsafe_allow_html=True)
        aggregate = benchmark_summary.get("aggregate", benchmark.get("aggregate", {}))
        render_insight_cards([
            {"title": "Search Time Gain", "value_label": "Average", "value": f"{aggregate.get('avg_search_time_gain_min', 0)} min", "note": "Agent mode reduction."},
            {"title": "Resilience Gain", "value_label": "Average", "value": aggregate.get("avg_resilience_gain", 0), "note": "Higher is better."},
            {"title": "Hotspot Reduction", "value_label": "Average", "value": aggregate.get("avg_hotspot_reduction", 0), "note": "Fewer critical SRM blocks."},
        ], columns=3)
        
        benchmark_frame = state_manager.get_benchmark_frame(benchmark, st.session_state.benchmark_toggle)
        chart = build_benchmark_chart(benchmark_frame)
        if chart is not None:
            st.plotly_chart(
                chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("benchmark-summary", step_number, str(st.session_state.benchmark_toggle)),
            )
            st.dataframe(benchmark_frame, width="stretch", hide_index=True)
        else:
            st.info(benchmark_summary.get("message", "Run the benchmark from the sidebar to generate a baseline comparison."))
            history_chart = build_kpi_chart(recent_states)
            if history_chart is not None:
                st.markdown("<div class='section-kicker'>Current Metrics History</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    history_chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=_chart_key("benchmark-history", step_number),
                )

        latest_chart = build_latest_baseline_chart(baseline_comparison)
        if latest_chart is not None:
            st.markdown("<div class='section-kicker'>Current Step Baseline</div>", unsafe_allow_html=True)
            st.plotly_chart(
                latest_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("benchmark-latest-baseline", step_number),
            )

    elif active_page == "Agent Loop":
        st.markdown("<div class='section-kicker'>Planner, Critic, Executor</div>", unsafe_allow_html=True)
        _agent_summary_cards(latest_result, goal, event_context)
        left, right = st.columns([1.05, 1])
        
        agent_df = state_manager.get_agent_frame(latest_result.get("agent_interactions", []), step_number)
        if not agent_df.empty:
            with left:
                _render_agent_decision_table(latest_result)
                if st.session_state.get("developer_mode", False):
                    with st.expander("🛠️ Open Developer Trace (Data Matrix)"):
                        st.dataframe(agent_df[["Agent", "Mode", "Action Taken", "Why"]], width="stretch", hide_index=True)
        else:
            with left:
                if agent_loop_steps:
                    st.info("Detailed agent interaction rows are pending; showing the step trace instead.")
                    step_trace_frame = pd.DataFrame([
                        {
                            "step": item.get("step", ""),
                            "output": item.get("output", ""),
                            "details": _format_reasoning_text(item.get("details", {})),
                        }
                        for item in agent_loop_steps
                    ])
                    st.dataframe(step_trace_frame, width="stretch", hide_index=True)
                else:
                    st.info("Run the system to populate the agent loop.")

        with right:
            st.markdown("<div class='section-kicker'>Intelligent Chain of Thought</div>", unsafe_allow_html=True)
            summarized_agents = _summarize_agent_rows(latest_result, agent_loop_steps)
            if summarized_agents:
                options = [item["agent"] for item in summarized_agents]
                current_agent = st.session_state.get("agent_loop_focus")
                if current_agent not in options:
                    current_agent = options[0]
                selected_agent = st.selectbox(
                    "Inspect agent",
                    options,
                    index=options.index(current_agent),
                    key="agent_loop_focus_selector",
                    on_change=_mark_ui_interaction,
                )
                st.session_state.agent_loop_focus = selected_agent
                selected_payload = next((item for item in summarized_agents if item["agent"] == selected_agent), summarized_agents[0])
                render_html_block(
                    f"""
                    <div class="focus-card">
                        <div class="focus-kicker">{selected_payload['agent']}</div>
                        <div class="focus-route">{selected_payload['action']}</div>
                        <div class="focus-reason">{selected_payload['why']}</div>
                        <div class="slot-timestamp">Latest signal: {selected_payload['signal'] or 'No extra signal recorded.'}</div>
                    </div>
                    """
                )
                quick_cols = st.columns(min(4, len(summarized_agents)))
                for column, item in zip(quick_cols, summarized_agents[:4]):
                    with column:
                        st.metric(item["agent"], item["action"][:28] + ("..." if len(item["action"]) > 28 else ""))
            else:
                st.info("Planner, critic, and executor trace will appear after the first simulation step.")

            if st.session_state.get("developer_mode", False):
                with st.expander("🛠️ Open raw JSON payload dumps"):
                    st.json({"planner_output": latest_result.get("planner_output", {}), "critic_output": latest_result.get("critic_output", {})})

        cycle_df = state_manager.get_cycle_frame(recent_cycles, step_number)
        if not cycle_df.empty:
            st.markdown("#### 📊 Agent Decision History (with LLM Tracking)")
            st.dataframe(cycle_df.tail(8)[["Step", "Event", "Planner", "Final", "Reward", "LLM Used", "LLM Influence", "LLM Decision"]], hide_index=True, width="stretch")
            llm_influence_total = cycle_df[cycle_df["LLM Influence"] == "🎯 Modified"].shape[0]
            if llm_influence_total > 0:
                st.success(f"🧠 LLM has directly modified **{llm_influence_total}** decision(s) in this session — hybrid intelligence confirmed.")
            if st.session_state.get("developer_mode", False):
                with st.expander("🛠️ Open raw JSON payload dumps"):
                    st.json({"planner_output": latest_result.get("planner_output", {}), "critic_output": latest_result.get("critic_output", {})})

    elif active_page == "Reasoning":
        st.markdown("<div class='section-kicker'>Gemini Budget & LLM Decision Log</div>", unsafe_allow_html=True)
        _render_decision_explainability(decision_explanation)
        if reasoning_summary:
            st.markdown(f"### 🤖 Agent Narrative Summary")
            st.success(f"**Current Strategy:** {reasoning_summary.get('fallback_label', 'Adaptive Response')}")
            
            # Big Narrative Block
            st.markdown(
                f"""
                <div style="background: rgba(142, 200, 255, 0.05); border-left: 5px solid #4da3ff; padding: 1.5rem; border-radius: 0 15px 15px 0; margin-bottom: 1.5rem;">
                    <div style="font-size: 1.15rem; line-height: 1.6; color: #e1e7f0; font-style: italic;">
                        "{reasoning_summary.get('reason', 'The agent is currently monitoring baseline network pressure.')}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols = st.columns(4)
            cols[0].metric("Final Decision", reasoning_summary.get("decision", "HOLD"))
            cols[1].metric("Agent Confidence", f"{float(reasoning_summary.get('confidence', 0))*100:.0f}%")
            cols[2].metric("Risk Assessment", reasoning_summary.get("critic_risk", "LOW").upper())
            cols[3].metric("Thinking Mode", reasoning_summary.get("budget_level", "LOCAL").replace("_", " ").upper())

            if st.session_state.get("developer_mode", False):
                alternatives = reasoning_summary.get("alternatives", [])
                if alternatives:
                    st.markdown("#### ⚖️ Alternative Strategies Evaluated (Dev Mode)")
                    for item in alternatives:
                        st.markdown(f"- {item}")
                
                schedule = latest_result.get("reasoning_budget", {}).get("signals", {})
                st.caption(
                    f"**Agentic Engine Pipeline:** Execution step {schedule.get('next_scheduled_llm_step', 0)} | "
                    f"Trigger Sensitivity: Escalated"
                )

        if llm_usage_summary:
            render_key_value_groups([
                {
                    "title": "Hybrid LLM Usage",
                    "items": [
                        {"label": "Total steps", "value": llm_usage_summary.get("total_steps", 0)},
                        {"label": "Gemini attempts", "value": llm_usage_summary.get("gemini_attempts", 0)},
                        {"label": "Forced attempts", "value": llm_usage_summary.get("forced_live_attempts", 0)},
                        {"label": "Gemini calls", "value": llm_usage_summary.get("gemini_calls", 0)},
                        {"label": "Remaining budget", "value": llm_usage_summary.get("remaining_budget", 0)},
                        {"label": "Gemini fallbacks", "value": llm_usage_summary.get("gemini_failures", 0)},
                        {"label": "Cache used", "value": llm_usage_summary.get("cache_used", 0)},
                        {"label": "Simulated Gemini", "value": llm_usage_summary.get("simulated_gemini", 0)},
                        {"label": "Local reasoning", "value": llm_usage_summary.get("local_reasoning", 0)},
                        {"label": "Last Gemini step", "value": llm_usage_summary.get("last_gemini_step", "-")},
                        {"label": "LLM influence", "value": f"{llm_usage_summary.get('llm_influence_pct', 0)}%"},
                    ],
                }
            ])
            if llm_usage_summary.get("budget_guard_active"):
                st.warning("Budget guard is active: Gemini usage reached the reserve threshold, so the runtime is holding the remaining calls for operator-triggered highlights.")
        if llm_status.get("quota_backoff", {}).get("active") and llm_status.get("quota_backoff", {}).get("kind") == "daily_quota":
            render_html_block(
                """
                <div class="quota-panel">
                    <strong>Why live Gemini is not visible right now</strong>
                    <span>The active project/key is still in daily quota exhaustion, so the 10-step heartbeat is being routed into simulated and local reasoning instead of a live cloud call. Execution should still continue through the agent loop without freezing.</span>
                </div>
                """
            )

        router_trace = llm_status.get("router_trace", [])
        active_route = llm_status.get("active_route", {})
        if router_trace or llm_status.get("router_mode"):
            st.markdown("**LLM Orchestrator**")
            render_key_value_groups([
                {
                    "title": "Routing Strategy",
                    "items": [
                        {"label": "Mode", "value": llm_status.get("router_mode", "Single-Key Gemini")},
                        {"label": "Configured keys", "value": llm_status.get("api_key_count", 0)},
                        {"label": "Model tiers", "value": len(llm_status.get("model_sequence", []))},
                        {"label": "Active model", "value": active_route.get("model", llm_status.get("model", "-"))},
                    ],
                }
            ])
            if router_trace:
                router_frame = pd.DataFrame([
                    {
                        "Key": item.get("key", "-"),
                        "Model": item.get("model", "-"),
                        "Status": item.get("status", "-"),
                        "Latency": item.get("latency_seconds", "-"),
                        "Reason": item.get("reason", ""),
                    }
                    for item in router_trace[-12:]
                ])
                st.dataframe(router_frame.iloc[::-1], width="stretch", hide_index=True)

        if last_llm_decision:
            st.markdown("**Most Recent Gemini Advisory**")
            st.success(
                f"Step {last_llm_decision.get('step', '-')} | "
                f"{last_llm_decision.get('mode', 'llm_advisory').replace('_', ' ').title()} | "
                f"{last_llm_decision.get('action_text', 'No action stored.')}"
            )
            st.caption(f"Timestamp: {_format_live_timestamp(last_llm_decision.get('timestamp'))}")
            if last_llm_decision.get("requested"):
                st.caption("This step explicitly requested live Gemini reasoning.")
            if last_llm_decision.get("source") == "gemini_failed_fallback":
                st.warning("Live Gemini was attempted on this step, but the planner fell back to local reasoning.")
            st.write(last_llm_decision.get("rationale", "No LLM rationale stored yet."))
            compare_cols = st.columns(3)
            compare_cols[0].metric("Local Decision", last_llm_decision.get("local_action_text", "No action stored."))
            compare_cols[1].metric("Gemini Suggestion", last_llm_decision.get("llm_action_text", "No action stored."))
            compare_cols[2].metric("Final Decision", last_llm_decision.get("final_action_text", last_llm_decision.get("action_text", "No action stored.")))
            changed_fields = last_llm_decision.get("changed_fields", [])
            if changed_fields:
                st.success(f"LLM influence: Modified {', '.join(changed_fields).replace('_', ' ')}")
            else:
                st.caption(f"LLM influence: {last_llm_decision.get('influence_label', 'Confirmed')} the local decision.")
        else:
            st.info("No Gemini advisory has been recorded yet. In Auto mode, the planner requests Gemini every 10 steps or earlier when queue >= 3, entropy > 3.5, risk > 70, or a decision conflict appears.")

        # --- LLM DECISION PANEL ---
        planner_out = latest_result.get("planner_output", {})
        critic_out = latest_result.get("critic_output", {})
        budget_out = latest_result.get("reasoning_budget", {})
        if not planner_out and not critic_out:
            st.info("Planner and critic outputs are not available yet. Run one step to populate reasoning details.")
        llm_used = planner_out.get("llm_advisory_used") or critic_out.get("llm_advisory_used") or planner_out.get("llm_requested")
        budget_level = budget_out.get("budget_level", "local_only")
        severe_triggers = budget_out.get("severe_triggers", [])
        moderate_triggers = budget_out.get("moderate_triggers", [])
        gate_notes = budget_out.get("gate_notes", [])

        planner_source = planner_out.get("llm_source", "deterministic")
        if llm_used:
            llm_rationale = planner_out.get("rationale", "") or (critic_out.get("critic_notes") or [""])[0]
            llm_strategy = planner_out.get("strategy", "")
            llm_action = planner_out.get("proposed_action", {})
            st.success("🤖 **Gemini was invoked on this step** — advisory merged into final decision.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Budget Level", budget_level.replace("_", " ").title())
            c2.metric("Decision Mode", planner_out.get("decision_mode", "llm_advisory").replace("_", " ").title())
            c3.metric("LLM Action", llm_action.get("action", "none").upper())
            st.markdown("**📋 Gemini Rationale**")
            st.info(llm_rationale or "Gemini advisory was applied — rationale embedded in planner output.")
            compare_cols = st.columns(3)
            compare_cols[0].metric("Local Suggestion", planner_out.get("local_decision_snapshot", {}).get("action", "none").upper())
            compare_cols[1].metric("Gemini Suggestion", planner_out.get("llm_decision_snapshot", {}).get("action", "none").upper())
            compare_cols[2].metric("Final Decision", planner_out.get("final_decision_snapshot", {}).get("action", llm_action.get("action", "none")).upper())
            st.caption(f"Gemini outcome: {planner_out.get('llm_decision_status', 'modified').replace('_', ' ').title()}")
            if llm_strategy and st.session_state.get("developer_mode", False):
                st.markdown(f"**Strategy adopted:** {llm_strategy}")
            if st.session_state.get("developer_mode", False):
                with st.expander("🛠️ Open full Gemini planner JSON"):
                    st.json({
                        "local_decision": planner_out.get("local_decision_snapshot", {}),
                        "gemini_suggestion": planner_out.get("llm_decision_snapshot", {}),
                        "final_decision": planner_out.get("final_decision_snapshot", {}),
                        "strategy": llm_strategy,
                        "rationale": llm_rationale,
                    })
        else:
            if planner_source in {"local_simulated", "simulated_edge_intelligence", "cached"}:
                st.info(f"⚡ **Reasoning core remained active** — `{planner_source}` guided the planner while live Gemini was unavailable. Budget: `{budget_level}`")
            else:
                st.warning(f"⚡ **Gemini skipped this step** — deterministic agents handled decision. Budget: `{budget_level}`")
            if gate_notes:
                for note in gate_notes:
                    st.caption(f"🔒 {note}")
            signals = budget_out.get("signals", {})
            render_key_value_groups([
                {
                    "title": "Gemini Schedule",
                    "items": [
                        {"label": "Next Gemini call", "value": signals.get("next_scheduled_llm_step", "-")},
                        {"label": "Steps until call", "value": signals.get("steps_until_next_llm", "-")},
                        {"label": "Trigger reason", "value": signals.get("llm_trigger_reason", "local")},
                        {"label": "Event trigger", "value": "Yes" if signals.get("event_trigger_due") else "No"},
                        {"label": "Calls used", "value": f"{signals.get('gemini_calls_today', 0)} / {signals.get('gemini_budget_limit', 18)}"},
                        {"label": "Decision conflict", "value": "Yes" if signals.get("decision_conflict") else "No"},
                    ],
                }
            ])
            action_out = latest_result.get("action", {})
            st.markdown("**Deterministic Decision Made:**")
            det_col1, det_col2 = st.columns(2)
            det_col1.metric("Action", action_out.get("action", "none").upper())
            det_col2.metric("Vehicles", action_out.get("vehicles", 0))
            st.info(action_out.get("reason", "Agent selected deterministic routing based on current state."))

        st.divider()
        st.markdown("<div class='section-kicker'>What Triggered This Budget Decision</div>", unsafe_allow_html=True)
        _render_reasoning_budget(budget_out)

        st.divider()
        st.markdown("<div class='section-kicker'>Full Decision Audit</div>", unsafe_allow_html=True)
        _render_decision_audit(latest_result)

        agent_df = state_manager.get_agent_frame(latest_result.get("agent_interactions", []), step_number)
        if not agent_df.empty:
            st.markdown("<div class='section-kicker'>Multi-Agent Interaction Trace</div>", unsafe_allow_html=True)
            st.dataframe(agent_df[["Agent", "Mode", "Why", "Key Output"]], width="stretch", hide_index=True)
            with st.expander("Open full agent payloads"):
                st.json(latest_result.get("agent_interactions", []))
        else:
            st.info("Run the system to inspect why each agent or model path was chosen.")


    elif active_page == "Memory & Goals":
        st.markdown("<div class='section-kicker'>Persistent Learning View</div>", unsafe_allow_html=True)
        _memory_summary_cards(metrics)
        _render_goal_status(goal, kpis)
        learning_profile = memory_summary.get("learning_profile", metrics.get("learning_profile", {}))
        recent_failures = learning_profile.get("recent_failures", [])
        learning_insight = learning_profile.get("latest_learning_insight", "No specific route patterns consolidated yet.")
        render_html_block(
            f"""
            <div class="learning-banner">
                <strong>Latest learning change</strong>
                <span>{learning_insight}</span>
            </div>
            """
        )
        patterns = memory_summary.get("patterns", [])
        if patterns:
            st.markdown("**Learned patterns**")
            for pattern in patterns:
                st.markdown(f"- {pattern}")
        
        if learning_profile:
            render_key_value_groups([
                {
                    "title": "Learning Signals",
                    "items": [
                        {"label": "Global transfer bias", "value": learning_profile.get("global_transfer_bias", "-")},
                        {"label": "Recent reward avg", "value": learning_profile.get("recent_reward_avg", "-")},
                        {"label": "Tracked failures", "value": len(recent_failures)},
                        {"label": "Blocked routes", "value": len(learning_profile.get("blocked_routes", []))},
                        {"label": "LLM memory rules", "value": len(learning_profile.get("llm_memory_rules", []))},
                    ],
                }
            ])
            trend_chart = build_performance_trend_chart(recent_states)
            if trend_chart is not None:
                st.markdown("<div class='section-kicker'>Learning Effect Over Recent Steps</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    trend_chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=_chart_key("memory-learning-trend", step_number),
                )
            goal_history = memory_summary.get("goal_history", [])
            if goal_history:
                with st.expander("Autonomous goal revisions", expanded=False):
                    goal_frame = pd.DataFrame([
                        {
                            "objective": item.get("objective", "-"),
                            "priority_zone": item.get("priority_zone", "-"),
                            "target_hotspots": item.get("target_congested_zones", "-"),
                            "target_search_time_min": item.get("target_search_time_min", "-"),
                            "status": item.get("status", "-"),
                            "revision_reason": item.get("revision_reason", "-"),
                            "revision_count": item.get("revision_count", 0),
                        }
                        for item in goal_history
                    ])
                    st.dataframe(goal_frame, width="stretch", hide_index=True)
            llm_rules = learning_profile.get("llm_memory_rules", [])
            if llm_rules:
                with st.expander("LLM-derived route rules", expanded=False):
                    st.dataframe(pd.DataFrame(llm_rules), width="stretch", hide_index=True)
        else:
            st.info("Learning profile is not available yet. Run one step to initialize memory signals.")
        history = memory_summary.get("history", [])
        if history:
            with st.expander("Recent decision memory", expanded=False):
                st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
        
        trace_frame = pd.DataFrame([
            {"Step": str(item.get("step")), "Mode": str(item.get("mode")), "Action": str(item.get("action"))[:60]}
            for item in snapshot.get("trace", [])
        ])
        if not trace_frame.empty:
            with st.expander("Open memory trace log", expanded=False):
                st.dataframe(trace_frame.tail(8), width="stretch", hide_index=True)

    elif active_page == "Notifications":
        st.markdown("<div class='section-kicker'>Proactive User Notifications</div>", unsafe_allow_html=True)
        _render_notifications(notification_summary.get("items", notifications))
        if alerts:
            st.markdown("<div class='section-kicker'>Real-Time App User Alerts</div>", unsafe_allow_html=True)
            alert_frame = pd.DataFrame([
                {
                    "Level": item.get("level", "info"),
                    "Alert": item.get("title", "SRM parking update"),
                    "Message": item.get("message", ""),
                    "Block": item.get("block", "-"),
                    "Audience": item.get("audience", "parking_app_users"),
                }
                for item in alerts
            ])
            st.dataframe(alert_frame, width="stretch", hide_index=True)

        # Show live sensor context even with no dispatched notifications
        st.divider()
        st.markdown("<div class='section-kicker'>Live Alert Context</div>", unsafe_allow_html=True)
        alert_cols = st.columns(3)
        alert_cols[0].metric("Queue Length", kpis.get("queue_length", 0))
        alert_cols[1].metric("SRM Block Hotspots", kpis.get("congestion_hotspots", 0))
        alert_cols[2].metric("Resilience Score", kpis.get("resilience_score", 100))

        # Denied entries trigger
        denied = kpis.get("denied_entries", 0)
        if denied > 0:
            st.error(f"**{denied} vehicle(s) were denied entry this step** — overflow risk is elevated. Agent is monitoring for redirect opportunity.")
        else:
            st.success("No denied entries this step — SRM parking blocks are absorbing arrivals normally.")

        dispatch = notification_summary.get("dispatch", snapshot.get("notification_dispatch", []))
        if dispatch:
            dispatch_frame = _group_notification_dispatch(dispatch)
            st.success(f"{len(dispatch)} recent notification delivery record(s) available from the runtime loop.")
            with st.expander("Open delivery audit log"):
                st.dataframe(dispatch_frame, width="stretch", hide_index=True)
        else:
            st.info("The notification service is live. New alerts appear here when queue pressure, blocked SRM blocks, denied entries, or redirect events occur.")

    elif active_page == "Vehicle Flow":
        st.markdown("<div class='section-kicker'>Vehicle Entry, Exit, and Redirect Timeline</div>", unsafe_allow_html=True)
        movement_frame = _build_movement_frame(movement_log)
        if movement_frame.empty:
            st.info("Vehicle flow history will appear after the next backend step.")
        else:
            top = movement_frame.iloc[-1].to_dict()
            render_key_value_groups([
                {
                    "title": "Latest Vehicle Movement",
                    "items": [
                        {"label": "Timestamp", "value": top.get("Timestamp", "-")},
                        {"label": "SRM Block", "value": top.get("SRM Block", "-")},
                        {"label": "Entries / Exits", "value": f"{top.get('Entries', 0)} / {top.get('Exits', 0)}"},
                        {"label": "Occupied After", "value": top.get("Occupied After", 0)},
                    ],
                },
                {
                    "title": "Vehicle Type Split",
                    "items": [
                        {"label": "Car In", "value": top.get("Car In", 0)},
                        {"label": "Bike In", "value": top.get("Bike In", 0)},
                        {"label": "Car Out", "value": top.get("Car Out", 0)},
                        {"label": "Bike Out", "value": top.get("Bike Out", 0)},
                    ],
                },
            ])
            redirect_frame = pd.DataFrame([
                {
                    "Timestamp": _format_live_timestamp(item.get("timestamp")),
                    "Step": item.get("step", "-"),
                    "Route": f"{item.get('from', '-')} → {item.get('to', '-')}",
                    "Total Redirected": item.get("vehicles", 0),
                    "Cars Redirected": item.get("car_vehicles", 0),
                    "Bikes Redirected": item.get("bike_vehicles", 0),
                    "Reason": item.get("reason", "Agent route execution"),
                }
                for item in actions
            ])
            if not redirect_frame.empty:
                st.markdown("<div class='section-kicker'>Redirect Execution History</div>", unsafe_allow_html=True)
                st.dataframe(redirect_frame.iloc[::-1], width="stretch", hide_index=True)
            st.markdown("<div class='section-kicker'>Full Vehicle Flow Log</div>", unsafe_allow_html=True)
            st.dataframe(movement_frame.iloc[::-1], width="stretch", hide_index=True)


    elif active_page == "AI Chat":
        st.markdown("<div class='section-kicker'>SRM Parking Operations Assistant</div>", unsafe_allow_html=True)
        if llm_status.get("quota_backoff", {}).get("active"):
            st.warning("Gemini chat is paused by quota backoff. Local chat fallback will still answer operational questions.")
        _render_assistant_briefing(assistant_briefing)
        suggestion_cols = st.columns(4)
        suggestions = ["Which SRM block is best right now?", "What event is affecting parking?", "Show the latest allocation decision", "Which block is most congested?"]
        for col, suggestion in zip(suggestion_cols, suggestions):
            with col:
                if st.button(suggestion, width="stretch"):
                    response = api_bridge.ask(suggestion)
                    st.session_state.chat_response = response["answer"]
                    st.session_state.chat_response_meta = response
                    
        with st.form("chat_form", clear_on_submit=False):
            raw_query = st.text_input("Ask about parking rush, best SRM blocks, current event impact, or dynamic allocation")
            submitted = st.form_submit_button("Ask")

        if submitted:
            cleaned_query = validator.sanitize_query(raw_query)
            if cleaned_query:
                with st.spinner("AI Generating..."):
                    response = api_bridge.ask(cleaned_query)
                    st.session_state.chat_response = response["answer"]
                    st.session_state.chat_response_meta = response
            else:
                st.session_state.chat_response = "Invalid input rejected."
                st.session_state.chat_response_meta = {"source": "local_validation", "llm_used": False, "reason": "Input validation rejected the query."}

        if st.session_state.chat_response:
            meta = st.session_state.chat_response_meta
            st.caption(f"Answer source: {meta.get('source', 'unknown')} | Gemini used: {'Yes' if meta.get('llm_used') else 'No'} | {meta.get('reason', '')}")
            st.success(st.session_state.chat_response)

    if st.session_state.run:
        _schedule_reload(speed)

if __name__ == "__main__":
    main()
