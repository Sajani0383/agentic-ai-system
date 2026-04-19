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
    if "chat_response" not in st.session_state:
        st.session_state.chat_response = ""
    if "chat_response_meta" not in st.session_state:
        st.session_state.chat_response_meta = {}
    if "benchmark_toggle" not in st.session_state:
        st.session_state.benchmark_toggle = False
    if "force_llm" not in st.session_state:
        st.session_state.force_llm = False

def _schedule_reload(seconds):
    time.sleep(max(0.2, seconds))
    st.rerun()

def _build_llm_summary(llm_status, is_forced=False):
    quota_backoff = llm_status.get("quota_backoff", {})
    if is_forced:
        return (
            "⚡ Strategic Overdrive Active",
            "System is strictly prioritizing live Gemini reasoning for mission-critical operations. Safety locks and quota cooldowns are currently overridden.",
            "success",
        )
    if quota_backoff.get("active"):
        return (
            "Efficient Local Mode Active",
            f"System has transitioned to Autonomous Edge Intelligence ({quota_backoff.get('remaining_seconds', 0)}s remaining in cloud optimization cycle). Resilient multi-agent heuristics are currently driving the parking policy.",
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
            f"Model: {llm_status.get('model', 'gemini')}. System is running in Hybrid Mode, dynamically balancing cloud-based LLM reasoning with efficient local heuristics.",
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

def _render_notifications(notifications):
    if not notifications:
        st.info("No active campus alerts right now.")
        return
    for notification in notifications:
        text = f"**{notification.get('title', 'Update')}**  \n{notification.get('message', '')}"
        level = notification.get("level")
        if level == "error": st.error(text)
        elif level == "warning": st.warning(text)
        elif level == "success": st.success(text)
        else: st.info(text)

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
        ("Reward", f"{reward_score:+.2f}", "Improved network balance."),
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
            "title": "Goal Status",
            "items": [
                {"label": "Status", "value": status},
                {"label": "Objective", "value": goal.get("objective", "-")},
                {"label": "Priority zone", "value": goal.get("priority_zone", "-")},
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
        st.success("The current step satisfies the active goal thresholds.")
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

def _render_decision_summary_block(latest_result, baseline_comparison):
    if not latest_result:
        return
    action = latest_result.get("action", {})
    act_type = action.get("action", "none").upper()
    reason_raw = action.get("reason", "Network pressure is within bounds.")
    
    # Condense the reason into a punchy one-liner if it's a redirect
    if act_type == "REDIRECT":
        source = action.get("from", "Unknown")
        target = action.get("to", "Unknown")
        vehicles = int(action.get("vehicles", 0))
        punchy_reason = f"{source} congested → {target} has free capacity → Redirecting {vehicles} vehicles now."
        display_type = "REDIRECT"
        bg_col = "rgba(77, 163, 255, 0.12)"
        border_col = "#4da3ff"
        text_col = "#cce5ff"
    else:
        # Build intelligent context from live state
        state = latest_result.get("state", {})
        if state:
            crowded = min(state, key=lambda z: state[z].get("free_slots", 0))
            best = max(state, key=lambda z: state[z].get("free_slots", 0))
            source = crowded
            target = best
        else:
            crowded, best, source, target = "-", "-", "-", "-"
        vehicles = 0
        punchy_reason = f"Agents analyzed all zones → {source} pressured but within safe thresholds → Holding for next cycle."
        display_type = "MONITORING"
        bg_col = "rgba(80, 200, 120, 0.07)"
        border_col = "#50c878"
        text_col = "#b0d8b8"

    st.markdown(
        f"""
        <div style="background: {bg_col}; border-left: 5px solid {border_col}; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div style="min-width: 120px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: {text_col};">Decision</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #fff;">{display_type}</div>
                </div>
                <div style="min-width: 120px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: {text_col};">Source Zone</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: #fff;">{source}</div>
                </div>
                <div style="min-width: 120px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: {text_col};">Destination</div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: #fff;">{target}</div>
                </div>
                <div style="min-width: 120px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: {text_col};">Volume</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #fff;">{vehicles}</div>
                </div>
            </div>
            <div style="margin-top: 1.5rem; font-size: 1.1rem; line-height: 1.4; color: #f5f5f5; font-style: italic; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem;">
                " {punchy_reason} "
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _render_llm_insight(latest_result):
    """Always-visible LLM decision panel — shows what Gemini decided OR why it was skipped."""
    if not latest_result:
        return
    critic = latest_result.get("critic_output", {})
    planner = latest_result.get("planner_output", {})
    budget = latest_result.get("reasoning_budget", {})
    llm_advisory_used = critic.get("llm_advisory_used") or planner.get("llm_advisory_used") or planner.get("llm_requested")
    budget_level = budget.get("budget_level", "local_only")
    planner_reason = budget.get("planner_reason", "")
    rationale = planner.get("rationale") or (critic.get("critic_notes") or [""])[0] or ""
    gate_notes = budget.get("gate_notes", [])

    if llm_advisory_used:
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
        action_desc = (
            f"Redirect {action.get('vehicles',0)} vehicles from {action.get('from','-')} → {action.get('to','-')}"
            if action.get("action") == "redirect"
            else "No redirect — system stable"
        )
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.02); border-left: 3px solid rgba(150,150,180,0.4); padding: 0.65rem 1rem; border-radius: 4px 10px 10px 4px; margin-bottom: 1rem;">
                <strong style="color: #8899bb; display: block; margin-bottom: 0.2rem;">⚡ Deterministic Step — Gemini Not Needed</strong>
                <span style="font-size: 0.85rem; color: #7a8fa6; display: block;">Decision: <b style='color:#ccd3d9'>{action_desc}</b></span>
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
            return f"Redirected {transferred} vehicle(s), prevented congestion escalation, and reduced pressure by {hotspot_delta:.0f} zone(s) while keeping search time stable."
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
    route = f"{action.get('from', '-') } -> {action.get('to', '-')}" if action.get("action") == "redirect" else "No route change"
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
                {"label": "Hotspot delta", "value": f"{baseline_comparison.get('hotspot_delta', 0):+.0f} zone(s)"},
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
                {"label": "Learning Note", "value": latest_result.get("reward", {}).get("adaptation_note", "Policy stable.")},
                {"label": "Memory Status", "value": "Route Avoided" if latest_result.get("planner_output", {}).get("memory_avoidance_triggered") else "Nominal"},
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
        perception_string = f"{crowded} congestion {occupancy}%"
    else:
        perception_string = "Network observation loaded"
        
    planner_route = f"Redirect {planner_action.get('vehicles', 0)} vehicles to {planner_action.get('to', '-')}" if planner_action.get("action") == "redirect" else "Maintain system baseline"
    critic_notes = critic.get("critic_notes", [])
    critic_reason = critic_notes[0] if critic_notes else f"Valid ({critic.get('risk_level', 'low')} risk)"
    critic_string = critic_reason if critic.get("approved") else f"Rejected: {critic_reason}"
    policy_string = "Approve" if action.get("action") == planner_action.get("action") else "Fallback to safety policy"
    
    if action.get("status") == "failed":
        action_string = f"Execution failed: {action.get('failure_reason', 'Network timeout')}"
        action_color = "#ff9d91"
    else:
        action_string = f"Redirect {action.get('vehicles', 0)} vehicles" if action.get("action") == "redirect" else "No redirect executed"
        action_color = "#4bd38a"

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; font-family: monospace; line-height: 1.8;">
        <div style="color: #4da3ff;"><b>Step 1: Perception</b> &rarr; <span style="color: #ccc;">{perception_string}</span></div>
        <div style="color: #4da3ff;"><b>Step 2: Planner</b> &rarr; <span style="color: #ccc;">Suggest {planner_route.lower()}</span></div>
        <div style="color: #4da3ff;"><b>Step 3: Critic</b> &rarr; <span style="color: #ccc;">{critic_string}</span></div>
        <div style="color: #4da3ff;"><b>Step 4: Policy</b> &rarr; <span style="color: #ccc;">{policy_string}</span></div>
        <div style="color: {action_color};"><b>Step 5: Action</b> &rarr; <span style="color: #fff;">{action_string}</span></div>
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
            "Zone": row["Zone"],
            "Status": status,
            "Free Slots": int(row["Free"]),
            "Utilisation": f"{utilisation:.1f}%",
        })
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
                {"label": "Critic Gemini requested", "value": "Yes" if critic.get("llm_requested") else "No"},
                {"label": "Critic Gemini changed review", "value": "Yes" if critic.get("llm_advisory_used") else "No"},
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

    with st.spinner("Synchronizing Simulation Core..."):
        snapshot = api_bridge.get_snapshot()

    if not snapshot:
        st.stop() # Stops execution gracefully if completely crashed

    current_scenario = snapshot.get("scenario_mode", "Auto Schedule")
    metrics = snapshot.get("metrics", {})
    step_number = metrics.get("steps", 0)

    st.sidebar.header("Simulation Controls")
    selected_scenario = st.sidebar.selectbox("Campus Scenario", SCENARIOS, index=SCENARIOS.index(current_scenario) if current_scenario in SCENARIOS else 0)
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
    )
    if new_force_llm != force_llm_active:
        if api_bridge.set_force_llm(new_force_llm):
            st.rerun()

    if st.sidebar.button("🔄 Reset AI Quota Cooldown", help="Clear local 429 backoffs and retry Gemini immediately."):
        api_bridge.reset_llm_runtime_state()
        st.toast("AI runtime state reset. Attempting fresh Gemini link...")
        st.rerun()

    llm_stride = snapshot.get("latest_result", {}).get("reasoning_budget", {}).get("signals", {}).get("llm_stride_steps", 10)
    st.sidebar.caption(f"Auto uses Gemini once every {llm_stride} steps when available. Demo allows one planner advisory when available. Local never calls Gemini.")

    st.session_state.run = st.sidebar.toggle("Autonomous Mode", value=st.session_state.run)
    speed = st.sidebar.slider("Step Interval (seconds)", 1.0, 8.0, 3.0)
    st.sidebar.caption("Use Run One Step first to test logic.")
    
    benchmark_episodes = st.sidebar.slider("Benchmark Episodes", 1, 5, 3)
    benchmark_steps = st.sidebar.slider("Benchmark Steps", 6, 15, 10)
    
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Run One Step", width="stretch"):
        with st.spinner("Tick..."):
            api_bridge.step_simulation()
        st.rerun()
        
    if c2.button("Pause", width="stretch"):
        st.session_state.run = False
        st.rerun()

    c3, c4 = st.sidebar.columns(2)
    if c3.button("Resume", width="stretch"):
        st.session_state.run = True
        st.rerun()
    if c4.button("Reset Runtime", width="stretch"):
        api_bridge.reset(clear_memory=False)
        st.session_state.run = False
        st.rerun()

    if st.sidebar.button("Run Benchmark", width="stretch"):
        with st.spinner("Benchmarking Agents vs Baseline..."):
            api_bridge.run_benchmark(benchmark_episodes, benchmark_steps)
            st.session_state.benchmark_toggle = not st.session_state.benchmark_toggle
        st.session_state.run = False
        st.rerun()

    st.sidebar.divider()
    st.session_state.developer_mode = st.sidebar.toggle("🖥️ Developer Mode", value=st.session_state.get('developer_mode', False))

    now = time.time()
    if st.session_state.run and now - st.session_state.last_run >= speed:
        st.session_state.last_run = now
        api_bridge.step_simulation()
        snapshot = api_bridge.get_snapshot() # Refresh locally during tick instead of full rerun cache miss
        metrics = snapshot.get("metrics", {})
        step_number = metrics.get("steps", 0)

    state = snapshot.get("state", {})
    latest_result = snapshot.get("latest_result", {})
    latest_transition = snapshot.get("latest_transition", {})
    goal = snapshot.get("goal", {})
    event_context = snapshot.get("event_context", {})
    kpis = snapshot.get("kpis", {})
    notifications = snapshot.get("notifications", [])
    recent_cycles = snapshot.get("recent_cycles", [])
    recent_states = snapshot.get("recent_states", [])
    llm_status = snapshot.get("llm_status", {})
    llm_mode = snapshot.get("llm_mode", "auto")
    benchmark = snapshot.get("benchmark", {})
    benchmark_summary = snapshot.get("benchmark_summary", {})
    assistant_briefing = snapshot.get("assistant_briefing", {})
    operational_signals = latest_result.get("operational_signals", latest_transition.get("dynamic_signals", {}))
    baseline_comparison = latest_result.get("baseline_comparison", {})
    reasoning_summary = snapshot.get("reasoning_summary", {})
    agent_loop_steps = snapshot.get("agent_loop_steps", [])
    memory_summary = snapshot.get("memory_summary", {})
    notification_summary = snapshot.get("notification_summary", {})
    last_llm_decision = snapshot.get("last_llm_decision", {})
    llm_usage_summary = snapshot.get("llm_usage_summary", {})

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
        st.sidebar.info("Quota optimization is active: scheduled Gemini checkpoint every 10 steps.")
    elif force_llm:
        st.sidebar.info("Overdrive mode overrides local gating. Every step will visibly attempt live Gemini reasoning.")

    # Cached heavy processing bound strictly to the step number cache key
    state_frame = state_manager.get_state_frame(state, step_number)

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
    m5.metric("Saved Steps", step_number)

    signal_cards(event_context, latest_result, kpis, goal)
    _render_system_status_bar(latest_result, event_context, kpis, llm_status, llm_mode)
    _render_decision_card(latest_result, baseline_comparison)

    if st.session_state.run:
        if speed < 3:
            st.warning("Autonomous mode is running quickly. For free-tier Gemini, 3-5 seconds or manual steps are safer.")
        else:
            st.info(f"Autonomous mode is active. The simulation refreshes every {speed:.1f} seconds.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["Operations", "Events & KPIs", "Benchmark", "Agent Loop", "Reasoning", "Memory & Goals", "Notifications", "AI Chat"]
    )

    with tab1:
        st.markdown("<div class='section-kicker'>Live Operations</div>", unsafe_allow_html=True)
        _render_decision_summary_block(latest_result, baseline_comparison)
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
                        {"label": "Congestion Hotspots", "value": f"{baseline.get('congestion_hotspots', 0)} → {agent_kpis.get('congestion_hotspots', 0)}"},
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

        st.markdown("<div class='section-kicker'>Zone Pressure</div>", unsafe_allow_html=True)
        st.dataframe(_build_zone_status_frame(state_frame), width="stretch", hide_index=True)

        left, right = st.columns([1, 1])
        with st.spinner("Plotting layouts..."): # Loading indicators injected natively
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

        with st.expander("Advanced View: zone cards, occupancy chart, and detailed table", expanded=False):
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
        lower_left.markdown("<div class='section-kicker'>Latest Zone Transition</div>", unsafe_allow_html=True)
        
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
                            {"label": "Recommended zone", "value": event_context.get("recommended_zone", "-")},
                            {"label": "Focus zone", "value": event_context.get("focus_zone", "-")},
                        ],
                    },
                ]
            )

    with tab2:
        st.markdown("<div class='section-kicker'>Industry Outcome Metrics</div>", unsafe_allow_html=True)
        render_insight_cards([
            {"title": "Search Time", "value_label": "Estimated", "value": f"{kpis.get('estimated_search_time_min', 0)} min", "note": "Lower is better."},
            {"title": "Space Utilisation", "value_label": "Current", "value": f"{kpis.get('space_utilisation_pct', 0)}%", "note": "Healthy utilisation."},
            {"title": "Allocation Success", "value_label": "Current", "value": f"{kpis.get('allocation_success_pct', 0)}%", "note": "Reallocation success."},
            {"title": "Congestion Hotspots", "value_label": "Current", "value": str(kpis.get('congestion_hotspots', 0)), "note": "Zones deeply congested."},
        ], columns=4)

        kpi_chart = build_kpi_chart(recent_states)
        if kpi_chart is not None:
            st.plotly_chart(
                kpi_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("events-kpi", step_number),
            )

    with tab3:
        st.markdown("<div class='section-kicker'>Simulation Proof</div>", unsafe_allow_html=True)
        aggregate = benchmark_summary.get("aggregate", benchmark.get("aggregate", {}))
        render_insight_cards([
            {"title": "Search Time Gain", "value_label": "Average", "value": f"{aggregate.get('avg_search_time_gain_min', 0)} min", "note": "Agent mode reduction."},
            {"title": "Resilience Gain", "value_label": "Average", "value": aggregate.get("avg_resilience_gain", 0), "note": "Higher is better."},
            {"title": "Hotspot Reduction", "value_label": "Average", "value": aggregate.get("avg_hotspot_reduction", 0), "note": "Fewer critical zones."},
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

        latest_chart = build_latest_baseline_chart(baseline_comparison)
        if latest_chart is not None:
            st.markdown("<div class='section-kicker'>Current Step Baseline</div>", unsafe_allow_html=True)
            st.plotly_chart(
                latest_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("benchmark-latest-baseline", step_number),
            )

    with tab4:
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
            left.info("Run the system to populate the agent loop.")

        with right:
            st.markdown("<div class='section-kicker'>Intelligent Chain of Thought</div>", unsafe_allow_html=True)
            
            # Show individual agent reasoning more prominently
            for agent_data in latest_result.get("agent_interactions", []):
                agent_name = agent_data.get("Agent", "Agent")
                with st.expander(f"🧠 {agent_name} Logic", expanded=True):
                    st.write(f"**Action:** {agent_data.get('Action Taken', 'none')}")
                    st.info(agent_data.get("Why") or "No reasoning recorded.")
                    if agent_data.get("Key Output"):
                        st.caption(f"Signal: {agent_data.get('Key Output')}")

            if st.session_state.get("developer_mode", False):
                with st.expander("🛠️ Open raw JSON payload dumps"):
                    st.json({"planner_output": latest_result.get("planner_output", {}), "critic_output": latest_result.get("critic_output", {})})

        cycle_df = state_manager.get_cycle_frame(recent_cycles, step_number)
        if not cycle_df.empty and st.session_state.get("developer_mode", False):
            with st.expander("🛠️ Open recent agent cycle history"):
                st.dataframe(styled_cycle_frame(cycle_df.tail(6)), width="stretch", hide_index=True)

    with tab5:
        st.markdown("<div class='section-kicker'>Gemini Budget & LLM Decision Log</div>", unsafe_allow_html=True)
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
                        {"label": "Gemini fallbacks", "value": llm_usage_summary.get("gemini_failures", 0)},
                        {"label": "Cache used", "value": llm_usage_summary.get("cache_used", 0)},
                        {"label": "Simulated Gemini", "value": llm_usage_summary.get("simulated_gemini", 0)},
                        {"label": "Local reasoning", "value": llm_usage_summary.get("local_reasoning", 0)},
                    ],
                }
            ])

        if last_llm_decision:
            st.markdown("**Most Recent Gemini Advisory**")
            st.success(
                f"Step {last_llm_decision.get('step', '-')} | "
                f"{last_llm_decision.get('mode', 'llm_advisory').replace('_', ' ').title()} | "
                f"{last_llm_decision.get('action_text', 'No action stored.')}"
            )
            if last_llm_decision.get("requested"):
                st.caption("This step explicitly requested live Gemini reasoning.")
            if last_llm_decision.get("source") == "gemini_failed_fallback":
                st.warning("Live Gemini was attempted on this step, but the planner fell back to local reasoning.")
            st.write(last_llm_decision.get("rationale", "No LLM rationale stored yet."))
        else:
            st.info("No Gemini advisory has been recorded yet. In Auto mode, the planner will request Gemini once every 10 steps when the provider is available.")

        # --- LLM DECISION PANEL ---
        planner_out = latest_result.get("planner_output", {})
        critic_out = latest_result.get("critic_output", {})
        budget_out = latest_result.get("reasoning_budget", {})
        llm_used = planner_out.get("llm_advisory_used") or critic_out.get("llm_advisory_used") or planner_out.get("llm_requested")
        budget_level = budget_out.get("budget_level", "local_only")
        severe_triggers = budget_out.get("severe_triggers", [])
        moderate_triggers = budget_out.get("moderate_triggers", [])
        gate_notes = budget_out.get("gate_notes", [])

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
            if llm_strategy and st.session_state.get("developer_mode", False):
                st.markdown(f"**Strategy adopted:** {llm_strategy}")
            if st.session_state.get("developer_mode", False):
                with st.expander("🛠️ Open full Gemini planner JSON"):
                    st.json({"proposed_action": llm_action, "strategy": llm_strategy, "rationale": llm_rationale})
        else:
            st.warning(f"⚡ **Gemini skipped this step** — deterministic agents handled decision. Budget: `{budget_level}`")
            if gate_notes:
                for note in gate_notes:
                    st.caption(f"🔒 {note}")
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


    with tab6:
        st.markdown("<div class='section-kicker'>Persistent Learning View</div>", unsafe_allow_html=True)
        _memory_summary_cards(metrics)
        _render_goal_status(goal, kpis)
        learning_profile = memory_summary.get("learning_profile", metrics.get("learning_profile", {}))
        recent_failures = learning_profile.get("recent_failures", [])
        learning_insight = learning_profile.get("latest_learning_insight", "No specific route patterns consolidated yet.")
        
        st.info(f"**Latest Learning Adaptation:** {learning_insight}")
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
                    ],
                }
            ])
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

    with tab7:
        st.markdown("<div class='section-kicker'>Proactive User Notifications</div>", unsafe_allow_html=True)
        _render_notifications(notification_summary.get("items", notifications))

        # Show live sensor context even with no dispatched notifications
        st.divider()
        st.markdown("<div class='section-kicker'>Live Alert Context</div>", unsafe_allow_html=True)
        alert_cols = st.columns(3)
        alert_cols[0].metric("Queue Length", kpis.get("queue_length", 0))
        alert_cols[1].metric("Congestion Hotspots", kpis.get("congestion_hotspots", 0))
        alert_cols[2].metric("Resilience Score", kpis.get("resilience_score", 100))

        # Denied entries trigger
        denied = kpis.get("denied_entries", 0)
        if denied > 0:
            st.error(f"**{denied} vehicle(s) were denied entry this step** — overflow risk is elevated. Agent is monitoring for redirect opportunity.")
        else:
            st.success("No denied entries this step — network is absorbing arrivals normally.")

        dispatch = notification_summary.get("dispatch", snapshot.get("notification_dispatch", []))
        if dispatch:
            dispatch_frame = _group_notification_dispatch(dispatch)
            st.success(f"{len(dispatch)} recent notification delivery record(s) available from the runtime loop.")
            with st.expander("Open delivery audit log"):
                st.dataframe(dispatch_frame, width="stretch", hide_index=True)
        else:
            st.info("The notification service is live. New alerts appear here when queue pressure, blocked zones, denied entries, or redirect events occur.")


    with tab8:
        st.markdown("<div class='section-kicker'>Parking Operations Assistant</div>", unsafe_allow_html=True)
        if llm_status.get("quota_backoff", {}).get("active"):
            st.warning("Gemini chat is paused by quota backoff. Local chat fallback will still answer operational questions.")
        _render_assistant_briefing(assistant_briefing)
        suggestion_cols = st.columns(4)
        suggestions = ["Which zone is best right now?", "What event is affecting parking?", "Show the latest allocation decision", "Which block is most congested?"]
        for col, suggestion in zip(suggestion_cols, suggestions):
            with col:
                if st.button(suggestion, width="stretch"):
                    response = api_bridge.ask(suggestion)
                    st.session_state.chat_response = response["answer"]
                    st.session_state.chat_response_meta = response
                    
        with st.form("chat_form", clear_on_submit=False):
            raw_query = st.text_input("Ask about parking rush, best zones, current event impact, or dynamic allocation")
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
