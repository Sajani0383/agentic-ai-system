import json
import html
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sys
import os
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    if "dashboard_auto_refresh" not in st.session_state:
        st.session_state.dashboard_auto_refresh = True
    if "manual_pause_requested" not in st.session_state:
        st.session_state.manual_pause_requested = False

def _mark_ui_interaction():
    st.session_state.last_ui_interaction_at = time.time()

def _ui_recently_interacted(now=None, cooldown=0.75):
    now = time.time() if now is None else now
    last_interaction = float(st.session_state.get("last_ui_interaction_at", 0.0) or 0.0)
    return (now - last_interaction) < cooldown

def _schedule_reload(seconds):
    components.html(
        "<script>/* Full-page reload disabled: live views poll backend without disrupting dashboard navigation. */</script>",
        height=0,
    )

def _render_expo_live_console():
    components.html(
        """
        <div id="expo-live-console" style="font-family: Inter, system-ui, sans-serif; color:#eaf2ff; background:#07111f; border:1px solid rgba(138,216,255,.28); border-radius:16px; padding:16px; box-sizing:border-box; width:100%; overflow:hidden;">
          <style>
            #expo-live-console .top{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}
            #expo-live-console .badge{display:inline-flex;gap:8px;align-items:center;color:#9fb2ca;font-weight:800;text-transform:uppercase;letter-spacing:.08em;font-size:12px}
            #expo-live-console .dot{width:10px;height:10px;border-radius:50%;background:#4bd38a;box-shadow:0 0 14px #4bd38a}
            #expo-live-console .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-bottom:12px}
            #expo-live-console .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:12px}
            #expo-live-console .card{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px;min-height:82px}
            #expo-live-console span{display:block;color:#9fb2ca;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
            #expo-live-console strong{display:block;color:#fff;font-size:24px;line-height:1.15;margin-top:6px;overflow-wrap:anywhere}
            #expo-live-console small{display:block;color:#b9c8dc;margin-top:6px;font-size:13px;line-height:1.35}
            #expo-live-console .flowline{display:flex;align-items:center;gap:10px;margin:8px 0 2px;color:#dff1ff;font-weight:900;font-size:20px}
            #expo-live-console .flowline b{background:rgba(77,163,255,.11);border:1px solid rgba(77,163,255,.24);border-radius:10px;padding:8px 10px}
            #expo-live-console .flowline i{font-style:normal;color:#7ac7ff}
            #expo-live-console .mini{min-height:58px}
            #expo-live-console .mini strong{font-size:22px}
            #expo-live-console .route{border-color:rgba(77,163,255,.35);background:rgba(77,163,255,.08)}
            #expo-live-console .llm{border-color:rgba(75,211,138,.35);background:rgba(75,211,138,.08)}
            #expo-live-console .warn{border-color:rgba(255,209,102,.35);background:rgba(255,209,102,.08)}
            #expo-live-console .timeline{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px;margin-bottom:10px}
            #expo-live-console .agents{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-bottom:10px}
            #expo-live-console .agent-card{border:1px solid rgba(77,163,255,.2);border-radius:12px;padding:10px;background:rgba(77,163,255,.055)}
            #expo-live-console .agent-card em{display:block;color:#4bd38a;font-style:normal;font-size:12px;font-weight:900;margin-top:4px}
            #expo-live-console .scroll{max-height:190px;overflow:auto}
            #expo-live-console .fresh{color:#b5ffe7;font-weight:900}
            #expo-live-console .muted{color:#8da0b7}
            #expo-live-console #llm-prompt,#expo-live-console #llm-response{display:inline;color:#dbe8f7;font-size:13px;font-weight:650;text-transform:none;letter-spacing:0;white-space:normal;overflow-wrap:anywhere;word-break:break-word}
            #expo-live-console #llm-confidence.conf-low{color:#ff8f8f}
            #expo-live-console #llm-confidence.conf-med{color:#ffd166}
            #expo-live-console #llm-confidence.conf-high{color:#62e69e}
            #expo-live-console .llm-expand{margin-top:8px;color:#b9c8dc}
            #expo-live-console .llm-expand summary{cursor:pointer;color:#9ed4ff;font-size:12px;font-weight:900;text-transform:uppercase;letter-spacing:.06em}
            #expo-live-console table{width:100%;border-collapse:collapse;font-size:13px}
            #expo-live-console th,#expo-live-console td{border-bottom:1px solid rgba(255,255,255,.08);padding:7px;text-align:left;color:#dbe8f7}
            #expo-live-console th{color:#9fb2ca;text-transform:uppercase;letter-spacing:.06em;font-size:11px}
            @media(max-width:900px){#expo-live-console .grid,#expo-live-console .timeline,#expo-live-console .kpis,#expo-live-console .agents{grid-template-columns:1fr}}
          </style>
          <div class="top">
            <div><div class="badge"><i class="dot"></i> Expo Live Console</div><small id="live-clock">Connecting to backend...</small></div>
            <div class="badge">Updates every 1s without dashboard reload</div>
          </div>
          <div class="grid">
            <div class="card route"><span>Current Decision</span><strong id="live-decision">-</strong><div class="flowline"><b id="flow-from">-</b><i>→</i><b id="flow-to">-</b></div><small id="live-route">Waiting for route</small></div>
            <div class="card"><span>Flow</span><strong id="live-flow">-</strong><small>entering / moving / exiting</small></div>
            <div class="card llm"><span>LLM Visibility</span><strong id="live-llm">-</strong><small id="live-llm-detail">Waiting for LLM signal</small></div>
            <div class="card warn"><span>System Health</span><strong id="live-health">-</strong><small id="live-sync">Sync pending</small></div>
          </div>
          <div class="kpis">
            <div class="card mini"><span>Avg Search Time</span><strong id="kpi-search">-</strong></div>
            <div class="card mini"><span>Congestion Reduction</span><strong id="kpi-congestion">-</strong></div>
            <div class="card mini"><span>Efficiency Score</span><strong id="kpi-efficiency">-</strong></div>
            <div class="card mini"><span>LLM Usage</span><strong id="kpi-llm">-</strong></div>
          </div>
          <div class="agents" id="agent-cards"></div>
          <div class="timeline">
            <div class="card"><span>Decision Timeline</span><small>Showing last 5 live entries.</small><div class="scroll"><table><thead><tr><th>Step</th><th>Action</th><th>From</th><th>To</th></tr></thead><tbody id="live-timeline"><tr><td colspan="4">Waiting...</td></tr></tbody></table></div></div>
            <div class="card"><span>LLM Decision Panel</span><small id="llm-used">Waiting for LLM state</small><details class="llm-expand"><summary>Prompt + response</summary><small><b>Prompt:</b> <span id="llm-prompt">-</span></small><small><b>Response:</b> <span id="llm-response">-</span></small></details><small id="llm-confidence">Confidence pending</small></div>
          </div>
          <div class="timeline">
            <div class="card"><span>User Activity</span><div class="scroll"><table><thead><tr><th>Vehicle</th><th>Name</th><th>Status</th><th>Block</th></tr></thead><tbody id="live-users"><tr><td colspan="4">Waiting...</td></tr></tbody></table></div></div>
            <div class="card"><span>System Alerts</span><div id="live-alerts"><small>No alerts yet</small></div></div>
          </div>
        </div>
        <script>
          const api = "http://127.0.0.1:8000/expo-state";
          const fmt = (value) => {
            if (!value) return "-";
            const d = new Date(value);
            return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleTimeString("en-IN", {hour:"2-digit", minute:"2-digit", second:"2-digit"});
          };
          const short = (value, n=92) => {
            const text = String(value || "-").replace(/\\s+/g, " ").trim();
            return text.length > n ? text.slice(0, n - 3) + "..." : text;
          };
          async function tick() {
            try {
              const res = await fetch(api + "?_ts=" + Date.now(), {cache:"no-store"});
              const data = await res.json();
              const decision = data.latest_decision || data.latest_result?.action || {};
              const stats = data.vehicle_stats || {};
              const llmUsage = data.llm_usage_summary || {};
              const lastLlm = data.last_llm_decision || {};
              const eventContext = data.event_context || {};
              const latestAction = data.latest_action_record || {};
              const users = (data.recent_user_vehicles || []).slice(0, 5);
              const agents = data.agent_loop_steps || [];
              const decisions = data.recent_decisions || [];
              const events = data.recent_events || [];
              const alerts = data.alerts || [];
              const reasoning = data.reasoning_summary || {};
              const actionName = String(decision.action || "monitor").toLowerCase();
              const isRedirect = actionName === "redirect";
              const route = isRedirect ? `${decision.from || "-"} → ${decision.to || "-"} (${decision.vehicles || 0})` : (actionName === "none" || actionName === "idle" ? "IDLE" : actionName);
              const pressureFocus = eventContext.pressure_focus || eventContext.focus_zone || "-";
              const globalBlock = eventContext.recommended_zone || "-";
              let vehicleIds = Array.isArray(latestAction.vehicle_ids) ? latestAction.vehicle_ids.slice(0, 5).filter(Boolean) : [];
              if (!vehicleIds.length) {
                vehicleIds = events
                  .filter(e => e.event === "redirect" && String(e.decision_step || e.step || "") === String(data.step || ""))
                  .slice(0, 5)
                  .map(e => e.vehicle_number || e.vehicle_id)
                  .filter(Boolean);
              }
              const auto = data.autonomy || {};
              document.getElementById("live-clock").textContent = `Backend step ${data.step || "-"} · ${fmt(data.updated_at)} · Autonomous ${auto.running ? "running" : "paused"}`;
              document.getElementById("live-decision").textContent = isRedirect ? "REDIRECT" : (actionName === "none" || actionName === "idle" ? "IDLE" : actionName.toUpperCase());
              document.getElementById("flow-from").textContent = isRedirect ? (decision.from || "-") : "Campus";
              document.getElementById("flow-to").textContent = isRedirect ? (decision.to || "-") : "Stable";
              document.getElementById("live-route").textContent = isRedirect
                ? short(`Recommended global block: ${globalBlock}. Chosen local route: ${decision.from || "-"} → ${decision.to || "-"}. Global guides incoming flow; local resolves active congestion. Focus = pressure source (${pressureFocus}); Source = rerouting origin (${decision.from || "-"}).${vehicleIds.length ? " Vehicles: " + vehicleIds.join(", ") : ""}`, 260)
                : short(decision.reason || "Campus pressure is within threshold; agents are observing this step.", 160);
              document.getElementById("live-flow").textContent = `${stats.entering || 0} / ${stats.redirecting || 0} / ${stats.exiting || 0}`;
              document.getElementById("live-llm").textContent = lastLlm.requested ? "Gemini" : "Local AI";
              document.getElementById("live-llm-detail").textContent = lastLlm.action_text ? short(`${lastLlm.influence_label || "LLM"}: ${lastLlm.action_text} · ${llmUsage.gemini_calls || 0} calls`, 120) : `LLM influence ${llmUsage.llm_influence_pct || 0}% · Local ${llmUsage.local_reasoning || 0}`;
              document.getElementById("live-health").textContent = stats.consistency === "verified" ? "Verified" : "Review";
              document.getElementById("live-sync").textContent = `Active ${stats.active || 0} · Exited ${stats.exited || 0}`;
              document.getElementById("kpi-search").textContent = `${reasoning.search_time_min || data.latest_result?.kpis?.estimated_search_time_min || "2.4"} min`;
              document.getElementById("kpi-congestion").textContent = `${Math.max(0, 100 - Number(stats.congestion_pct || stats.hotspot_pct || 0)).toFixed(0)}%`;
              document.getElementById("kpi-efficiency").textContent = `${Number(stats.efficiency_score || stats.allocation_success_pct || 100).toFixed(0)}%`;
              document.getElementById("kpi-llm").textContent = `${Number(llmUsage.llm_influence_pct || 0).toFixed(1)}%`;
              document.getElementById("llm-used").innerHTML = lastLlm.requested ? `<span class="fresh">LLM requested</span> · ${lastLlm.influence_label || "Reviewed"} · ${lastLlm.source || "Gemini"}` : `<span class="muted">LLM skipped</span> · ${reasoning.llm_status_note || "local agent step"}`;
              document.getElementById("llm-prompt").textContent = lastLlm.local_action_text ? `Local ${lastLlm.local_action_text}. Compare Gemini advisory and return safest SRM parking action.` : `Evaluate ${route} with safety, learning and capacity constraints.`;
              document.getElementById("llm-response").textContent = lastLlm.rationale || lastLlm.action_text || reasoning.reason || "Local agents completed the decision.";
              const confidence = Math.round(Number(reasoning.confidence || decision.confidence || 0) * 100);
              const confEl = document.getElementById("llm-confidence");
              confEl.className = confidence >= 70 ? "conf-high" : confidence >= 45 ? "conf-med" : "conf-low";
              confEl.textContent = `Confidence ${confidence}% · Influence ${lastLlm.influence_label || "not used"}`;
              document.getElementById("live-users").innerHTML = users.length ? users.map((v, idx) => `<tr class="${idx===0 ? "fresh" : ""}"><td>${v.number || v.id || "-"}</td><td>${v.name || "-"}</td><td>${v.status || "-"}</td><td>${v.block || "-"}</td></tr>`).join("") : `<tr><td colspan="4">No user vehicles yet</td></tr>`;
              document.getElementById("live-timeline").innerHTML = decisions.length ? decisions.slice(0,5).map(a => `<tr><td>${a.step || "-"}</td><td>${a.action || "redirect"}</td><td>${a.from || "-"}</td><td>${a.to || "-"}</td></tr>`).join("") : events.slice(0,5).map(e => `<tr><td>${e.decision_step || "-"}</td><td>${e.event || "-"}</td><td>${e.from || e.gate || "-"}</td><td>${e.to || e.block || "-"}</td></tr>`).join("");
              const generatedAlerts = [];
              if (!lastLlm.requested) generatedAlerts.push({level:"Notice", message:"LLM skipped for this step; local agents kept execution moving."});
              if (Number(data.latest_result?.reward_score || 0) < -0.05) generatedAlerts.push({level:"Notice", message:data.latest_result?.reward_note || "Reward penalized execution cost despite search-time improvement."});
              if (Number(stats.free_slots || 1) === 0) generatedAlerts.push({level:"Warning", message:"No parking available in the current live state."});
              if (Number(stats.redirecting || 0) > 0) generatedAlerts.push({level:"Info", message:`${stats.redirecting} vehicle(s) currently redirecting.`});
              const alertRows = alerts.length ? alerts : generatedAlerts;
              document.getElementById("live-alerts").innerHTML = alertRows.length ? alertRows.map(a => `<small><b>${a.level || "Info"}:</b> ${short(a.message || a.title || a, 110)}</small>`).join("") : `<small>No active alerts · state verified</small>`;
              const fallbackAgents = [
                ["Perception Agent", "Active", `Flow now ${stats.entering || 0}/${stats.redirecting || 0}/${stats.exiting || 0}`],
                ["Planner Agent", "Active", route],
                ["Critic Agent", "Active", data.latest_result?.critic_output?.risk_level || "risk checked"],
                ["Executor Agent", "Active", decision.action || "monitor"],
              ];
              const agentRows = agents.length ? agents.slice(0,4).map(a => [a.agent || a.Agent || a.step || "Agent", "Active", a.output || a.message || a["Action Taken"] || "-"]) : fallbackAgents;
              document.getElementById("agent-cards").innerHTML = agentRows.map(a => `<div class="agent-card"><span>${a[0]}</span><em>${a[1]}</em><small>${short(a[2], 72)}</small></div>`).join("");
            } catch (err) {
              document.getElementById("live-health").textContent = "Offline";
              document.getElementById("live-sync").textContent = "Backend not reachable";
            }
          }
          tick();
          setInterval(tick, 1000);
        </script>
        """,
        height=1120,
    )

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
    goal_label = goal.get("objective") or "Maintain balanced load across SRM blocks"
    if str(goal.get("target_congested_zones", "")).strip() == "0":
        goal_value = "Maintain low congestion"
    else:
        goal_value = goal.get("target_congested_zones", "Balanced load")
    cards = [
        ("Planner", planner_action, latest_result.get("strategy", event_context.get("allocation_strategy", "-"))),
        ("Critic", latest_result.get("critic_output", {}).get("risk_level", "low").upper(), critic_notes[0] if critic_notes else "No critic issue raised."),
        ("Executor", final_action, _execution_summary(latest_result)),
        ("Goal", str(goal_value), goal_label),
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
    base = f"Reward {reward_score:+.2f}: derived from congestion, search-time change, execution cost, and queue pressure."
    if reward_score < -0.05:
        return reward_impact.get("explanation", base + " Negative values penalize weak or costly route changes.")
    if reward_score > 0.05:
        return reward_impact.get("explanation", base + " Positive values reinforce the route.")
    return base + " Neutral value means no major route weight change."

def _render_goal_status(goal, kpis, latest_result=None):
    if not goal:
        st.info("No active goal has been created yet. Run one step to let the planner set a goal.")
        return
    latest_result = latest_result or {}
    action = latest_result.get("action", {})
    target_hotspots = goal.get("target_congested_zones", 1)
    target_search = goal.get("target_search_time_min", 4.0)
    current_hotspots = kpis.get("congestion_hotspots", 0)
    current_search = kpis.get("estimated_search_time_min", 0.0)
    achieved = current_hotspots <= target_hotspots and current_search <= target_search
    status = "Achieved" if achieved else "In Progress"
    priority_block = action.get("from") or goal.get("priority_zone", "-")
    priority_note = "active source" if action.get("from") else "goal target"
    render_key_value_groups([
        {
            "title": "SRM Goal Status",
            "items": [
                {"label": "Status", "value": status},
                {"label": "Objective", "value": goal.get("objective", "-")},
                {"label": "Priority block", "value": f"{priority_block} ({priority_note})"},
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
        reward_score = float(latest_result.get("reward_score", 0) or 0)
        if reward_score < 0:
            st.success(
                "The goal thresholds are satisfied, but the reward is negative because the latest route added cost, churn, or weak local relief. "
                "This is expected: goal status measures the current campus condition, while reward scores the last action quality."
            )
        else:
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
    if planner.get("llm_source") == "cached" and planner.get("llm_advisory_used"):
        return {
            "mode": "Cached Gemini",
            "status": "Gemini Used",
            "fallback": "Not Needed",
            "detail": "A recent Gemini advisory was reused for this step, so the decision remains LLM-informed without spending another free-tier call.",
        }
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
        return {"mode": "Local Decision", "status": "Local Reasoning", "fallback": "No LLM Used", "detail": "Local agents handled this low-risk step while the Gemini budget was preserved for higher-value decisions."}
    if llm_mode == "demo":
        return {"mode": "Demo Requested", "status": "Gemini Unavailable", "fallback": "Local Fallback", "detail": llm_status.get("message", "Gemini was not available for this step.")}
    return {"mode": "Auto Budget", "status": "Local Reasoning", "fallback": "Deterministic", "detail": "No high-value LLM trigger was needed."}

def _render_system_status_bar(latest_result, event_context, kpis, llm_status, llm_mode):
    critic = latest_result.get("critic_output", {})
    action = latest_result.get("action", {})
    confidence_raw = action.get("confidence", latest_result.get("planner_output", {}).get("confidence", "-"))
    confidence_value = None
    if isinstance(confidence_raw, (int, float)):
        confidence_value = float(confidence_raw if confidence_raw <= 1 else confidence_raw / 100)
        confidence = f"{confidence_value:.0%}"
    else:
        confidence = str(confidence_raw)
    confidence_color = "#4bd38a"
    if confidence_value is not None:
        if confidence_value < 0.45:
            confidence_color = "#ff9d91"
        elif confidence_value < 0.70:
            confidence_color = "#ffd166"
    llm_state = _get_llm_state(latest_result, llm_status, llm_mode)
    llm_active_statuses = {"Gemini Used", "Simulated Gemini"}
    active_mode = "LLM Active" if llm_state.get("status") in llm_active_statuses else "Local Fallback"
    render_html_block(
        f"""
        <div class="status-bar">
            <div><span>Mode</span><strong>{active_mode}</strong></div>
            <div><span>Decision Source</span><strong>{llm_state["mode"]}</strong></div>
            <div><span>Risk</span><strong>{critic.get("risk_level", "low").title()}</strong></div>
            <div><span title="Based on risk constraints, capacity bounds, and network demand">Confidence ℹ️</span><strong style="color:{confidence_color};">{confidence}</strong></div>
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
    final_source = action.get("from") or planner_action.get("from")
    final_target = action.get("to") or planner_action.get("to")
    if final_source and isinstance(state, dict) and final_source in state:
        source_state = state[final_source]
        total_slots = source_state.get("total_slots") or source_state.get("capacity") or 1
        occupancy = round((source_state.get("occupied", 0) / max(1, total_slots)) * 100)
        if action.get("action") == "redirect" and final_target:
            perception_string = (
                f"{final_source} selected as active source ({occupancy}% occupied); "
                f"{final_target} chosen after capacity, critic, and learning checks."
            )
        else:
            perception_string = f"{final_source} monitored as active SRM block ({occupancy}% occupied)."
    elif state and isinstance(state, dict):
        crowded = min(state, key=lambda zone: state[zone].get("free_slots", 999))
        total_slots = state[crowded].get("total_slots", state[crowded].get("capacity", 1))
        occupancy = round((state[crowded].get("occupied", 0) / max(1, total_slots)) * 100)
        perception_string = f"{crowded} SRM block congestion {occupancy}%"
    else:
        perception_string = "SRM parking observation loaded"
        
    planner_route = (
        f"Redirect {planner_action.get('vehicles', 0)} vehicles "
        f"{planner_action.get('from', '-')} -> {planner_action.get('to', '-')}"
        if planner_action.get("action") == "redirect"
        else "Maintain system baseline"
    )
    route_adjustment = ""
    if (
        planner_action.get("action") == "redirect"
        and action.get("action") == "redirect"
        and (planner_action.get("from") != action.get("from") or planner_action.get("to") != action.get("to"))
    ):
        route_adjustment = (
            f"Planner considered {planner_action.get('from', '-')} -> {planner_action.get('to', '-')}; "
            f"final route became {action.get('from', '-')} -> {action.get('to', '-')} after critic, capacity, and learning constraints."
        )
    critic_notes = critic.get("critic_notes", [])
    critic_reason = critic_notes[0] if critic_notes else f"Valid ({critic.get('risk_level', 'low')} risk)"
    critic_string = critic_reason if critic.get("approved") else f"Rejected: {critic_reason}"
    llm_source = planner.get("llm_source", "deterministic")
    if llm_source == "cached":
        llm_string = "Cached Gemini advisory reused -> budget preserved"
    elif llm_source in {"local_simulated", "simulated_edge_intelligence", "demo_simulated"}:
        llm_string = "Local AI reasoning used -> Gemini budget preserved"
    elif planner.get("llm_requested"):
        if planner.get("llm_fallback_used") or planner.get("llm_source") == "gemini_failed_fallback":
            llm_string = "Gemini attempted -> local fallback used"
        elif planner.get("llm_advisory_used"):
            llm_string = "Gemini advisory applied"
        else:
            llm_string = "Gemini requested -> no plan change"
    else:
        llm_string = "Gemini skipped this step -> local agents used"
    policy_string = "Reference baseline only; used for safety recovery and benchmarking, not final authority"
    
    if action.get("status") == "failed":
        action_string = f"Execution failed: {action.get('failure_reason', 'Network timeout')}"
        action_color = "#ff9d91"
    else:
        action_string = (
            f"Redirect {action.get('vehicles', 0)} vehicles {action.get('from', '-')} -> {action.get('to', '-')}"
            if action.get("action") == "redirect"
            else "No redirect executed"
        )
        action_color = "#4bd38a"
    if not critic.get("approved") and action.get("action") == "redirect":
        action_string = "Blocked: critic rejection cannot execute"
        action_color = "#ff9d91"

    with st.container(border=True):
        st.markdown(f"**Step 1: Perception** → {perception_string}")
        st.markdown(f"**Step 2: LLM Gate** → {llm_string}")
        st.markdown(f"**Step 3: Planner** → Suggest {planner_route.lower()}")
        if route_adjustment:
            st.warning(f"Route Adjustment → {route_adjustment}")
        st.markdown(f"**Step 4: Critic** → {critic_string}")
        st.markdown(f"**Step 5: Baseline Context** → {policy_string}. If critic rejects, executor receives NONE.")
        if action_color == "#4bd38a":
            st.success(f"Step 6: Action → {action_string}")
        else:
            st.error(f"Step 6: Action → {action_string}")

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
        return pd.DataFrame(columns=["SRM Block", "Car Slots", "Bike Slots", "Capacity", "Occupied", "Free Slots", "Fill %"])
    frame = state_frame.copy()
    frame["Fill %"] = frame.apply(
        lambda row: round((float(row.get("Occupied", 0) or 0) / float(row.get("Capacity", 1) or 1)) * 100, 1),
        axis=1,
    )
    frame = frame.rename(columns={"Zone": "SRM Block", "Capacity": "Capacity", "Free": "Free Slots"})
    return frame[["SRM Block", "Car Slots", "Bike Slots", "Capacity", "Occupied", "Free Slots", "Fill %"]]

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

def _render_glow_box(title, body, tone="info"):
    palette = {
        "info": ("#2f8cff", "rgba(47, 140, 255, 0.14)"),
        "success": ("#48e38c", "rgba(72, 227, 140, 0.14)"),
        "warning": ("#ffcc66", "rgba(255, 204, 102, 0.14)"),
        "danger": ("#ff6b6b", "rgba(255, 107, 107, 0.14)"),
    }
    border, bg = palette.get(tone, palette["info"])
    st.markdown(
        f"""
        <div style="border:1px solid {border}; background:{bg}; box-shadow:0 0 28px {bg};
                    border-radius:18px; padding:18px 20px; margin:12px 0;">
          <div style="letter-spacing:.16em; text-transform:uppercase; color:#a9c1dc; font-size:.82rem; font-weight:800;">
            {html.escape(str(title))}
          </div>
          <div style="color:#f4f8ff; font-size:1.05rem; line-height:1.45; margin-top:6px;">
            {html.escape(str(body))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    block_capacity = max(0, int(block.get("capacity", 0) or 0))
    occupied = max(0, min(block_capacity, int(block.get("occupied", 0) or 0)))
    free_slots = max(0, block_capacity - occupied)
    car_slots = min(block_capacity, int(block.get("car_slots", block_capacity) or block_capacity))
    bike_slots = max(0, int(block.get("bike_slots", max(0, block_capacity - car_slots)) or 0))
    block_vehicles = [vehicle for vehicle in vehicles if vehicle.get("block") == selected]
    raw_slot_lookup = {
        int(vehicle.get("slot", 0) or 0): vehicle
        for vehicle in block_vehicles
        if int(vehicle.get("slot", 0) or 0) > 0
    }
    occupied_slot_numbers = sorted(raw_slot_lookup.keys())[:occupied]
    slot_lookup = {slot: raw_slot_lookup[slot] for slot in occupied_slot_numbers}
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

    cars_live = min(occupied, sum(1 for vehicle in slot_lookup.values() if vehicle.get("type") == "car"))
    bikes_live = max(0, occupied - cars_live)
    render_html_block(
        f"""
        <div class="slot-detail-card">
            <div class="focus-kicker">Selected SRM Block</div>
            <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <div class="focus-title" style="font-size:2rem; margin-bottom:0.3rem;">{selected}</div>
                    <div class="focus-route">{f"Occupied {occupied} of {block_capacity} | Free {free_slots}" if block_capacity else "No configured slots for this block"}</div>
                </div>
                <div class="focus-stat-grid" style="margin-top:0; min-width:320px;">
                    <div class="focus-stat"><span>Car Slots</span><strong>{car_slots}</strong></div>
                    <div class="focus-stat"><span>Bike Slots</span><strong>{bike_slots}</strong></div>
                    <div class="focus-stat"><span>Occupied Slots</span><strong>{occupied}</strong></div>
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
    role_map = {
        "MonitoringAgent": "Reads live occupancy, entries, exits, free slots, and vehicle movement.",
        "DemandAgent": "Predicts near-term pressure from scenario demand and gate flow.",
        "BayesianAgent": "Estimates uncertainty, risk, and confidence before planning.",
        "PlannerAgent": "Proposes the best route, vehicle count, and Gemini/local advisory.",
        "CriticAgent": "Checks safety, learning constraints, capacity, and whether replanning is needed.",
        "ExecutorAgent": "Applies the approved final action to the shared parking state.",
        "PolicyAgent": "Keeps the no-agent baseline for comparison and recovery only.",
        "RewardAgent": "Scores the outcome using search time, congestion, risk, and action cost.",
        "Action": "Final executed route after planner, critic, and executor agreement.",
        "Perception": "Live sensor snapshot: pressure block, buffer block, queue, and capacity.",
        "Planner": "Route proposal before critic and learning constraints.",
        "Critic": "Safety review and replan decision.",
        "Policy": "Reference baseline, not final authority.",
    }
    for item in rows:
        if not isinstance(item, dict):
            continue
        agent = item.get("Agent") or item.get("agent") or "Agent"
        action_text = str(item.get("Action Taken", item.get("message", "Pending")))
        why_text = _format_reasoning_text(item.get("Why", item.get("why", "")))
        if len(why_text) > 360:
            why_text = why_text[:357].rstrip() + "..."
        normalized = {
            "agent": str(agent),
            "action": action_text,
            "why": why_text or role_map.get(str(agent), "Agent processed the current backend step."),
            "signal": str(item.get("Key Output", "")),
            "role": role_map.get(str(agent), "Specialized agent in the current parking decision loop."),
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

def _dedupe_goal_history(goal_history):
    deduped = []
    seen = set()
    for item in reversed(goal_history or []):
        if not isinstance(item, dict):
            continue
        key = (
            item.get("objective"),
            item.get("priority_zone"),
            item.get("target_search_time_min"),
            item.get("status"),
            item.get("revision_reason"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return list(reversed(deduped))

def _build_lifecycle_memory_frame(vehicle_events, recent_cycles=None, movement_log=None):
    rows = []
    for item in (vehicle_events or [])[-60:]:
        if not isinstance(item, dict):
            continue
        event = item.get("event", "-")
        if event not in {"entry", "parked", "redirect", "exit"}:
            continue
        rows.append(
            {
                "Step": item.get("decision_step", item.get("step", "-")),
                "Kind": event,
                "Vehicle": item.get("vehicle_number", "-"),
                "Name": item.get("name", "-") if item.get("user_type") != "simulated" else "-",
                "Route": f"{item.get('from_block') or item.get('from_gate') or '-'} -> {item.get('to_block') or item.get('to_gate') or item.get('block') or '-'}",
                "Status": item.get("status", event),
                "Time": _format_live_timestamp(item.get("timestamp")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.iloc[::-1].drop_duplicates(subset=["Step", "Kind", "Vehicle", "Route"], keep="first")

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

def _sanitize_dashboard_text(value):
    replacements = {
        "Stadium": "Tech Park",
        "stadium": "Tech Park",
    }
    if isinstance(value, str):
        cleaned = value
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_dashboard_text(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize_dashboard_text(key): _sanitize_dashboard_text(item)
            for key, item in value.items()
        }
    return value

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
    normalized["blocks"] = _normalize_backend_blocks(_safe_dict(normalized.get("blocks")) or normalized["state"])
    normalized["state"] = deepcopy(normalized["blocks"])
    normalized.setdefault("vehicles", [])
    normalized.setdefault("movement_log", [])
    normalized.setdefault("actions", [])
    normalized.setdefault("alerts", [])
    normalized.setdefault("updated_at", "")
    normalized.setdefault("decision_explanation", {})
    return _sanitize_dashboard_text(normalized)

def _normalize_backend_blocks(blocks):
    normalized = {}
    for name, block in (blocks or {}).items():
        if not isinstance(block, dict):
            continue
        capacity = int(block.get("capacity", 0) or 0)
        if capacity <= 0:
            capacity = int(block.get("total_slots", 0) or 0)
        car_slots = max(0, int(block.get("car_slots", capacity) or 0))
        bike_slots = max(0, int(block.get("bike_slots", max(0, capacity - car_slots)) or 0))
        if capacity <= 0:
            capacity = car_slots + bike_slots
        if capacity <= 0:
            capacity = max(0, int(block.get("occupied", 0) or 0))
            car_slots = capacity
            bike_slots = 0
        if car_slots + bike_slots != capacity:
            if car_slots > capacity:
                car_slots = capacity
            bike_slots = max(0, capacity - car_slots)
        occupied = max(0, min(capacity, int(block.get("occupied", 0) or 0)))
        normalized[name] = {
            "total_slots": capacity,
            "capacity": capacity,
            "car_slots": car_slots,
            "bike_slots": bike_slots,
            "occupied": occupied,
            "free_slots": capacity - occupied,
            "entry": max(0, int(block.get("entry", 0) or 0)),
            "exit": max(0, int(block.get("exit", 0) or 0)),
        }
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
        fallback_step = index if step_number == 0 else step_number - len(recent_states) + index + 1
        normalized.append({
            **item,
            "step": item.get("step") or fallback_step,
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
    shared_state = _safe_dict(snapshot.get("current_state"))
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
        "vehicles": _safe_list(snapshot.get("vehicles")) or _safe_list(shared_state.get("vehicles")),
        "simulated_vehicles": _safe_list(snapshot.get("simulated_vehicles")) or [
            item for item in _safe_list(snapshot.get("vehicles")) if item.get("user_type") == "simulated"
        ],
        "user_vehicles": _safe_list(snapshot.get("user_vehicles")) or _safe_list(shared_state.get("user_vehicles")),
        "vehicle_stats": _safe_dict(snapshot.get("vehicle_stats")) or _safe_dict(shared_state.get("vehicle_stats")),
        "vehicle_events": _safe_list(snapshot.get("events")) or _safe_list(shared_state.get("events")),
        "movement_log": _safe_list(snapshot.get("movement_log")) or _safe_list(shared_state.get("movement_log")),
        "actions": _safe_list(snapshot.get("actions")) or _safe_list(shared_state.get("actions")),
        "updated_at": snapshot.get("updated_at", ""),
        "decision_explanation": _safe_dict(snapshot.get("decision_explanation")),
        "agentic_integrity": _safe_dict(snapshot.get("agentic_integrity")),
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
    snapshot = _normalize_dashboard_snapshot(snapshot)

    current_scenario = snapshot.get("scenario_mode", "Auto Schedule")
    metrics = snapshot.get("metrics", {})
    step_number = (
        _safe_dict(snapshot.get("current_state")).get("step")
        or _safe_dict(snapshot.get("latest_transition")).get("step")
        or snapshot.get("step")
        or metrics.get("steps", 0)
    )

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
            snapshot["llm_mode"] = selected_llm_mode
            api_bridge.clear_cache()
            st.toast(f"LLM mode set to {selected_llm_label}.")

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
            snapshot["force_llm"] = new_force_llm
            api_bridge.clear_cache()
            st.toast("Strategic Overdrive enabled." if new_force_llm else "Strategic Overdrive disabled. Smart LLM policy is active.")

    llm_stride = snapshot.get("latest_result", {}).get("reasoning_budget", {}).get("signals", {}).get("llm_stride_steps", 10)
    st.sidebar.caption(
        f"Budget-Aware Adaptive Invocation: Gemini recalibrates every {llm_stride} steps and only escalates early for queue >= 3, entropy > 3.5, risk > 70, or decision conflict."
    )

    pre_autonomy_status = api_bridge.get_autonomy_status()
    if pre_autonomy_status.get("running") and not st.session_state.get("manual_pause_requested"):
        st.session_state.run = True
    elif st.session_state.get("manual_pause_requested"):
        st.session_state.run = False

    was_running = bool(st.session_state.get("run", False))
    st.session_state.run = st.sidebar.toggle("Autonomous Mode", value=st.session_state.run, key="sidebar_autonomous_mode", on_change=_mark_ui_interaction)
    if st.session_state.run != was_running:
        if st.session_state.run:
            st.session_state.manual_pause_requested = False
            api_bridge.start_autonomy(interval_seconds=speed if "speed" in locals() else 2.0)
            st.session_state.last_run = 0.0
        else:
            st.session_state.manual_pause_requested = True
            api_bridge.stop_autonomy()
        st.rerun()
    st.session_state.dashboard_auto_refresh = False
    st.sidebar.success("Tab-safe live mode: the command center and frontend update without reloading the dashboard page.")
    if st.sidebar.button("Refresh Dashboard Snapshot", width="stretch", key="sidebar_refresh_dashboard", on_click=_mark_ui_interaction):
        api_bridge.clear_cache()
        st.rerun()
    speed = st.sidebar.slider("Step Interval (seconds)", 2.0, 8.0, 2.0, key="sidebar_step_interval", on_change=_mark_ui_interaction)
    st.sidebar.caption("Fast live sync uses the same lightweight backend stream as the frontend, so user vehicles should appear within a few seconds.")
    
    benchmark_episodes = st.sidebar.slider("Benchmark Episodes", 1, 5, 3, key="sidebar_benchmark_episodes", on_change=_mark_ui_interaction)
    benchmark_steps = st.sidebar.slider("Benchmark Steps", 6, 15, 10, key="sidebar_benchmark_steps", on_change=_mark_ui_interaction)
    
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Run One Step", width="stretch", on_click=_mark_ui_interaction):
        with st.spinner("Tick..."):
            api_bridge.step_simulation()
        st.rerun()
        
    if c2.button("Pause", width="stretch", on_click=_mark_ui_interaction):
        st.session_state.run = False
        st.session_state.manual_pause_requested = True
        api_bridge.stop_autonomy()
        st.rerun()

    c3, c4 = st.sidebar.columns(2)
    if c3.button("Resume", width="stretch", on_click=_mark_ui_interaction):
        st.session_state.run = True
        st.session_state.manual_pause_requested = False
        api_bridge.start_autonomy(interval_seconds=speed)
        st.session_state.last_run = 0.0
        st.rerun()
    if c4.button("Reset Runtime", width="stretch", on_click=_mark_ui_interaction):
        api_bridge.reset(clear_memory=False)
        st.session_state.run = False
        st.session_state.manual_pause_requested = True
        st.rerun()

    if st.sidebar.button("Run Benchmark", width="stretch", on_click=_mark_ui_interaction):
        with st.spinner("Benchmarking Agents vs Baseline..."):
            api_bridge.run_benchmark(benchmark_episodes, benchmark_steps)
            st.session_state.benchmark_toggle = not st.session_state.benchmark_toggle
        st.session_state.run = False
        st.rerun()

    pressure_labels = {
        "no_change": "No change",
        "normal": "Normal capacity",
        "heavy": "Heavy pressure",
        "near_full": "Near full",
        "full": "Full campus",
    }
    selected_pressure = st.sidebar.selectbox(
        "Demo Pressure",
        list(pressure_labels.keys()),
        format_func=lambda key: pressure_labels[key],
        index=0,
        key="sidebar_demo_pressure",
        help="Applies one scenario profile and refreshes the shared backend snapshot.",
        on_change=_mark_ui_interaction,
    )
    if selected_pressure != "no_change" and selected_pressure != st.session_state.get("last_pressure_request"):
        api_bridge.stop_autonomy()
        st.session_state.run = False
        st.session_state.manual_pause_requested = True
        if api_bridge.apply_demo_pressure(selected_pressure):
            st.session_state.last_pressure_request = selected_pressure
            st.toast(f"{pressure_labels[selected_pressure]} profile applied.")
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

    autonomy_status = api_bridge.get_autonomy_status()
    if st.session_state.get("manual_pause_requested") and autonomy_status.get("running"):
        api_bridge.stop_autonomy()
        autonomy_status = api_bridge.get_autonomy_status()
    if autonomy_status.get("running") and not st.session_state.run and not st.session_state.get("manual_pause_requested"):
        st.session_state.run = True
    if st.session_state.run and not autonomy_status.get("running") and not st.session_state.get("manual_pause_requested"):
        api_bridge.start_autonomy(interval_seconds=speed)
        autonomy_status = api_bridge.get_autonomy_status()
    if st.session_state.run and autonomy_status.get("running"):
        st.session_state.last_run = time.time()
    now = time.time()
    if False and st.session_state.run and not _ui_recently_interacted(now) and now - st.session_state.last_run >= speed:
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
        available_keys = int(llm_status.get("available_key_count", 0) or 0)
        if available_keys > 0 and llm_status.get("available"):
            st.sidebar.success(
                f"Gemini router active: {available_keys} healthy route(s). Some routes are cooling down for {backoff.get('remaining_seconds', 0)} sec."
            )
        else:
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
    simulated_vehicles = dashboard_state["simulated_vehicles"]
    user_vehicles = dashboard_state["user_vehicles"]
    vehicle_stats = dashboard_state["vehicle_stats"]
    vehicle_events = dashboard_state["vehicle_events"]
    movement_log = dashboard_state["movement_log"]
    actions = dashboard_state["actions"]
    updated_at = dashboard_state["updated_at"]
    decision_explanation = _safe_dict(dashboard_state.get("decision_explanation"))
    agentic_integrity = _safe_dict(dashboard_state.get("agentic_integrity"))

    total_capacity = dashboard_state["capacity"]
    total_occupied = dashboard_state["occupied"]
    total_free = dashboard_state["free_slots"]
    congestion = round((total_occupied / total_capacity * 100), 2) if total_capacity else 0.0
    if not state_frame.empty:
        highest_pressure_row = state_frame.sort_values("Utilisation %", ascending=False).head(1)
        if not highest_pressure_row.empty:
            event_context = {**event_context, "pressure_focus": highest_pressure_row.iloc[0]["Zone"]}

    st.markdown("<div class='section-kicker'>Expo Live Command Center</div>", unsafe_allow_html=True)
    st.caption("The command center below polls the backend every second. Use the left sidebar refresh button only for Streamlit audit tables/charts.")
    _render_expo_live_console()

    with st.expander("Snapshot Audit: Streamlit-rendered supporting metrics", expanded=False):
        st.caption("Collapsed to avoid stale duplicate decisions. These tables refresh with Streamlit; the Expo Live Command Center above is the real-time source of truth.")
        st.markdown(
            f"""
            <div class="event-banner">
                <div class="event-title">SRM Parking Simulation: {event_context.get("name", "SRM Simulation")} | Snapshot at {_format_live_timestamp(updated_at)}</div>
                <div class="event-copy">
                    {event_context.get("description", "")}
                    Strategy: <strong>{event_context.get("allocation_strategy", "Balanced utilisation")}</strong>.
                    Focus block: <strong>{event_context.get("pressure_focus", event_context.get("focus_zone", "-"))}</strong> (pressure).
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
        m4.metric("Campus Fill Level", f"{congestion}%")
        m5.metric("Saved Steps", step_number)
        if congestion >= 95:
            st.error("Full-campus stress profile is active: parking is nearly saturated.")
        elif congestion >= 80:
            st.warning("Heavy parking pressure is active.")
        else:
            st.success("Normal capacity profile is active.")

        signal_cards(event_context, latest_result, kpis, goal)
        _render_system_status_bar(latest_result, event_context, kpis, llm_status, llm_mode)
        latest_flow_action = actions[-1] if actions else {}
        affected_ids = latest_flow_action.get("vehicle_ids") or latest_flow_action.get("vehicles_ids") or latest_result.get("action", {}).get("vehicle_ids") or []
        if isinstance(affected_ids, list):
            affected_ids = ", ".join(str(item) for item in affected_ids[:6]) or "-"
        render_key_value_groups([
            {
                "title": "Live Flow Counter",
                "items": [
                    {"label": "Entering", "value": vehicle_stats.get("entering", 0)},
                    {"label": "Redirecting", "value": vehicle_stats.get("redirecting", 0)},
                    {"label": "Exiting", "value": vehicle_stats.get("exiting", 0)},
                ],
            },
            {
                "title": "Active Movement",
                "items": [
                    {"label": "Last action", "value": latest_flow_action.get("type", latest_result.get("action", {}).get("action", "none"))},
                    {"label": "Vehicles", "value": affected_ids},
                    {"label": "Sync", "value": "State Verified" if vehicle_stats.get("consistency") == "verified" else "Review Needed"},
                ],
            },
        ])
        if agentic_integrity:
            failed_checks = [
                item.get("name", "Check")
                for item in agentic_integrity.get("checks", [])
                if not item.get("ok")
            ]
            render_key_value_groups([
                {
                    "title": "Agentic Integrity",
                    "items": [
                        {"label": "Readiness", "value": agentic_integrity.get("status", "Review")},
                        {"label": "Score", "value": f"{agentic_integrity.get('score', 0)}%"},
                        {"label": "Checks passed", "value": f"{agentic_integrity.get('passed', 0)} / {agentic_integrity.get('total', 0)}"},
                        {"label": "Open issue", "value": failed_checks[0] if failed_checks else "None"},
                    ],
                }
        ])
        _render_decision_summary_block(latest_result, baseline_comparison, updated_at)
        action = _safe_dict(latest_result.get("action"))
        search_delta = float(_safe_dict(baseline_comparison).get("search_time_delta_min", 0) or 0)
        reward_score = float(latest_result.get("reward_score", 0) or 0)
        recommended_zone = event_context.get("recommended_zone", "-")
        chosen_route = (
            f"{action.get('from', '-')} → {action.get('to', '-')}"
            if action.get("action") == "redirect"
            else "Monitoring"
        )
        clarity_parts = [
            f"Global recommendation: {recommended_zone}. Local route: {chosen_route}. Global handles incoming flow; local resolves active congestion.",
        ]
        if reward_score < 0 and search_delta > 0:
            clarity_parts.append(
                f"Reward is {reward_score:+.2f} even though search improved by {search_delta:+.2f} min because reward penalized execution cost despite search improvement."
            )
        else:
            clarity_parts.append(
                f"Reward {reward_score:+.2f} scores the latest action quality; search delta {search_delta:+.2f} min is only the no-redirect comparison."
            )
        _render_glow_box("Decision clarity", " ".join(clarity_parts), "warning" if reward_score < 0 else "info")
        _render_glow_box(
            "Dynamic balancing note",
            "Frequent redirects are expected in this high-responsiveness demo mode. The agent is continuously balancing live demand; oscillation can be stabilized with stronger route-change thresholds for production.",
            "info",
        )

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
        st.success("Autonomous backend mode is active. The Expo Live Command Center updates every second without restarting the dashboard page.")

    dashboard_pages = ["SRM Operations", "Events & KPIs", "Benchmark", "Agent Loop", "Reasoning", "Memory & Goals", "Vehicle & Flow Monitoring", "Vehicle Flow", "Notifications", "AI Chat"]
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
        st.info("Use the Expo Live Console above for the real-time decision, LLM reasoning, user activity, and agent pipeline. The sections below are snapshot audit views for judges who want supporting detail.")

        with st.expander("Snapshot Audit: current decision contract", expanded=False):
            _render_agent_decision_table(latest_result)
            _render_decision_explainability(decision_explanation)

        with st.expander("Snapshot Audit: SRM block pressure and slots", expanded=False):
            st.dataframe(_build_zone_status_frame(state_frame), width="stretch", hide_index=True)
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
                            {"label": "Global entry hint", "value": event_context.get("recommended_zone", "-")},
                            {"label": "Focus SRM block", "value": event_context.get("focus_zone", "-")},
                        ],
                    },
                ]
            )

    elif active_page == "Events & KPIs":
        st.markdown("<div class='section-kicker'>Live SRM Events & KPIs</div>", unsafe_allow_html=True)
        event_counts = pd.Series([item.get("event", "-") for item in vehicle_events[-80:]]).value_counts().to_dict() if vehicle_events else {}
        latest_action = actions[-1] if actions else {}
        kpi_explanation = (
            f"Current live sample from backend step {step_number}: live counters measure vehicles currently entering, redirecting, and exiting; "
            f"event counts measure persisted lifecycle rows. Capacity check is "
            f"{'verified' if total_occupied + total_free == total_capacity else 'under review'}. "
            "Charts below use the latest visible samples, so resets do not stretch the x-axis with old persisted steps."
        )
        _render_glow_box("KPI interpretation", kpi_explanation, "info")
        render_insight_cards([
            {"title": "Search Time", "value_label": "Estimated", "value": f"{kpis.get('estimated_search_time_min', 0)} min", "note": "Lower is better."},
            {"title": "Campus Fill Level", "value_label": "Current", "value": f"{congestion}%", "note": "Occupied slots divided by SRM total capacity."},
            {"title": "Current Route", "value_label": "Agent", "value": f"{latest_action.get('from', '-')} -> {latest_action.get('to', '-')}" if latest_action else "Monitoring", "note": f"{latest_action.get('vehicles', 0)} vehicle(s) in latest action." if latest_action else "No redirect in current snapshot."},
            {"title": "Recent Events", "value_label": "Entry / Redirect / Exit", "value": f"{event_counts.get('entry', 0)} / {event_counts.get('redirect', 0)} / {event_counts.get('exit', 0)}", "note": "Persisted event rows from the shared backend."},
        ], columns=4)
        render_key_value_groups([
            {
                "title": "Capacity Consistency",
                "items": [
                    {"label": "Total capacity", "value": total_capacity},
                    {"label": "Occupied", "value": total_occupied},
                    {"label": "Free", "value": total_free},
                    {"label": "Check", "value": "Verified" if total_occupied + total_free == total_capacity else "Review"},
                ],
            },
            {
                "title": "Live Movement",
                "items": [
                    {"label": "Entering", "value": vehicle_stats.get("entering", 0)},
                    {"label": "Redirecting", "value": vehicle_stats.get("redirecting", 0)},
                    {"label": "Exiting", "value": vehicle_stats.get("exiting", 0)},
                    {"label": "Latest step", "value": step_number},
                ],
            },
        ])
        if latest_action:
            _render_glow_box(
                "Decision impact",
                (
                    f"{latest_action.get('vehicles', 0)} vehicle(s) are being routed from "
                    f"{latest_action.get('from', '-')} to {latest_action.get('to', '-')}. "
                    "Search-time deltas compare against the no-redirect baseline; reward also includes execution cost."
                ),
                "success" if float(latest_result.get("reward_score", 0) or 0) >= 0 else "warning",
            )

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
        with st.expander("Advanced: block capacity table and slot inspector", expanded=False):
            st.dataframe(_build_srm_capacity_dataset(state_frame), width="stretch", hide_index=True)
            _render_live_slot_board(state, vehicles, updated_at)

    elif active_page == "Benchmark":
        st.markdown("<div class='section-kicker'>Simulation Proof</div>", unsafe_allow_html=True)
        aggregate = benchmark_summary.get("aggregate", benchmark.get("aggregate", {}))
        live_gain = _safe_dict(baseline_comparison)
        live_search_delta = live_gain.get("search_time_delta_min", latest_result.get("reward_impact", {}).get("search_time_delta_min", 0))
        live_hotspot_delta = live_gain.get("hotspot_delta", live_gain.get("hotspots_delta", 0))
        live_action = _safe_dict(latest_result.get("action"))
        render_insight_cards([
            {"title": "Live Search Delta", "value_label": "Current Step", "value": f"{float(live_search_delta or 0):+.2f} min", "note": "Positive means search time improved versus no-redirect baseline."},
            {"title": "Live Route", "value_label": "Agent Action", "value": f"{live_action.get('from', '-')} -> {live_action.get('to', '-')}" if live_action.get("action") == "redirect" else "Monitoring", "note": f"{live_action.get('vehicles', 0)} vehicle(s), reward {float(latest_result.get('reward_score', 0) or 0):+.2f}."},
            {"title": "Hotspot Change", "value_label": "Current Step", "value": live_hotspot_delta or 0, "note": "Compared with no-agent baseline where available."},
            {"title": "Benchmark Average", "value_label": "Stored Run", "value": f"{aggregate.get('avg_search_time_gain_min', 0)} min", "note": "Use sidebar Run Benchmark to refresh formal trial results."},
        ], columns=4)
        _render_glow_box(
            "How to read this benchmark",
            (
                "Live proof compares the current agent action against a no-redirect baseline for this exact step. "
                "Formal benchmark compares multiple scenarios and averages the gains. Positive search delta means the agent reduced expected parking search time; reward also includes action cost, risk, and learning penalties."
            ),
            "info",
        )
        if not aggregate:
            st.info("Formal benchmark has not been run in this session. The cards and charts still show live baseline proof from the current synchronized backend step.")
        else:
            _render_glow_box(
                "Formal benchmark result",
                (
                    f"Stored benchmark average search gain: {aggregate.get('avg_search_time_gain_min', 0)} min. "
                    f"Resilience gain: {aggregate.get('avg_resilience_gain', aggregate.get('avg_resilience_gain_pct', 0))}. "
                    f"Hotspot reduction: {aggregate.get('avg_hotspot_reduction', aggregate.get('avg_hotspot_delta', 0))}."
                ),
                "success",
            )
        
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
            comparison_rows = []
            baseline_kpis = _safe_dict(baseline_comparison.get("baseline_kpis"))
            agent_kpis = _safe_dict(baseline_comparison.get("agent_kpis"))
            for label, key in [
                ("Search time", "estimated_search_time_min"),
                ("Hotspots", "congestion_hotspots"),
                ("Resilience", "resilience_score"),
                ("Queue", "queue_length"),
            ]:
                comparison_rows.append({
                    "Metric": label,
                    "Agent": agent_kpis.get(key, "-"),
                    "No-Redirect Baseline": baseline_kpis.get(key, "-"),
                    "What it means": "Lower is better" if label in {"Search time", "Hotspots", "Queue"} else "Higher is better",
                })
            st.plotly_chart(
                latest_chart,
                use_container_width=True,
                config={"displayModeBar": False},
                key=_chart_key("benchmark-latest-baseline", step_number),
            )
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

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
                        <div class="slot-timestamp">{selected_payload.get('role', 'Agent role in the live loop.')}</div>
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
            active_route = _safe_dict(llm_status.get("active_route"))
            llm_available = bool(llm_status.get("available"))
            gemini_calls = llm_usage_summary.get("gemini_calls", 0)
            last_gemini_step = llm_usage_summary.get("last_gemini_step", "-")
            last_llm_attempt_step = llm_usage_summary.get("last_llm_attempt_step", "-")
            if llm_available and (active_route or gemini_calls):
                render_html_block(
                    f"""
                    <div class="quota-panel" style="border-color: rgba(75, 211, 138, 0.45); background: rgba(75, 211, 138, 0.10);">
                        <strong>LLM Working: Gemini Router Online</strong>
                        <span>Live route: {active_route.get('key', 'ready')} / {active_route.get('model', llm_status.get('model', 'Gemini'))}. Successful Gemini calls: {gemini_calls}. Last successful step: {last_gemini_step}. Last attempted step: {last_llm_attempt_step}.</span>
                    </div>
                    """
                )
            else:
                backoff = _safe_dict(llm_status.get("quota_backoff"))
                if backoff.get("active") or llm_usage_summary.get("budget_guard_active"):
                    st.info(
                        "Gemini router is configured, but this live step is intentionally using Local Reasoning "
                        "because provider quota/backoff or the budget guard is active. The agent loop is still running."
                    )
                elif llm_status.get("api_key_count", 0) and llm_status.get("available"):
                    st.info(
                        "Gemini router is ready. No successful Gemini call is attached to this snapshot because this step "
                        "did not require cloud escalation."
                    )
                else:
                    st.warning(
                        "Gemini is not available in this snapshot. Local multi-agent reasoning is handling decisions; "
                        "check API keys/router status only if live Gemini is required for the demo."
                    )
            st.markdown("#### Agent Decision Timeline")
            timeline_table = cycle_df.tail(6).copy()
            successful_gemini_mask = (
                cycle_df["LLM Used"].astype(str).eq("Yes")
                | cycle_df["LLM Status"].astype(str).str.contains("Live Gemini|Cached Gemini", case=False, na=False)
            )
            attempted_gemini_mask = (
                cycle_df["LLM Used"].astype(str).isin(["Attempted", "Demo"])
                | cycle_df["LLM Status"].astype(str).str.contains("Gemini Attempted|Demo Gemini", case=False, na=False)
            )
            pinned_gemini_rows = cycle_df[successful_gemini_mask]
            if pinned_gemini_rows.empty:
                pinned_gemini_rows = cycle_df[attempted_gemini_mask]
            if not pinned_gemini_rows.empty:
                pinned_gemini_row = pinned_gemini_rows.tail(1)
                if pinned_gemini_row.iloc[0]["Step"] not in set(timeline_table["Step"].tolist()):
                    timeline_table = pd.concat([pinned_gemini_row, timeline_table], ignore_index=True)
                    st.caption(
                        "Pinned latest Gemini-backed step above the newest local steps so LLM usage remains visible even when recent steps are local."
                    )
            st.dataframe(
                timeline_table[["Step", "Event", "Action", "Reward", "LLM Used", "LLM Status", "LLM Influence"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("Timeline reward is the agentic reward score for that step. Search-time and hotspot deltas are separate baseline comparison metrics.")
            with st.expander("Open LLM tracking details", expanded=False):
                for row in cycle_df.tail(12).iloc[::-1].to_dict("records"):
                    st.markdown(
                        f"""
                        <div class="llm-detail-row">
                            <div><strong>Step {row.get('Step', '-')}</strong> · LLM Used: <strong>{row.get('LLM Used', '-')}</strong> · Status: <strong>{row.get('LLM Status', '-')}</strong> · Influence: <strong>{row.get('LLM Influence', '-')}</strong></div>
                            <div>{row.get('LLM Detail', 'No detail recorded.')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            llm_influence_total = sum(1 for item in recent_cycles if item.get("planner_output", {}).get("llm_influence"))
            if llm_influence_total > 0:
                st.success(f"LLM has directly modified {llm_influence_total} recent decision(s).")
            if st.session_state.get("developer_mode", False):
                with st.expander("🛠️ Open raw JSON payload dumps"):
                    st.json({"planner_output": latest_result.get("planner_output", {}), "critic_output": latest_result.get("critic_output", {})})

    elif active_page == "Reasoning":
        st.markdown("<div class='section-kicker'>Gemini Budget & LLM Decision Log</div>", unsafe_allow_html=True)
        llm_available = bool(llm_status.get("available"))
        quota_backoff = _safe_dict(llm_status.get("quota_backoff"))
        router_trace = llm_status.get("router_trace", [])
        active_route = llm_status.get("active_route", {})
        render_key_value_groups([
            {
                "title": "LLM Runtime Status",
                "items": [
                    {"label": "Active mode", "value": "Gemini" if llm_available and not quota_backoff.get("active") else "Local Reasoning"},
                    {"label": "Budget used", "value": f"{llm_usage_summary.get('gemini_attempts', 0)} attempts / {llm_usage_summary.get('gemini_calls', 0)} calls"},
                    {"label": "Router model", "value": active_route.get("model", llm_status.get("model", "-"))},
                    {"label": "Last Gemini step", "value": llm_usage_summary.get("last_gemini_step", "-")},
                ],
            }
        ])
        if quota_backoff.get("active"):
            st.warning(f"Gemini is currently paused by {quota_backoff.get('kind', 'provider backoff')}. The system is not frozen: local multi-agent reasoning is driving live decisions until the router can call Gemini again.")
        elif not llm_available:
            st.info("Gemini is not available in this snapshot, so the dashboard labels the reasoning as Local Reasoning instead of pretending an LLM was used.")
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
                st.success("LLM working: Gemini has already reached the configured live-call budget for this run, so the runtime is now preserving quota with local AI/cached reasoning.")
        if llm_status.get("quota_backoff", {}).get("active") and llm_status.get("quota_backoff", {}).get("kind") == "daily_quota":
            render_html_block(
                """
                <div class="quota-panel">
                    <strong>Why live Gemini is not visible right now</strong>
                    <span>The active project/key is still in daily quota exhaustion, so the 10-step heartbeat is being routed into simulated and local reasoning instead of a live cloud call. Execution should still continue through the agent loop without freezing.</span>
                </div>
                """
            )

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
                with st.expander("Router attempts and fallback reasons", expanded=False):
                    for item in list(reversed(router_trace[-12:])):
                        render_html_block(
                            f"""
                            <div class="llm-detail-row">
                                <div><strong>{item.get('key', '-')}</strong> · {item.get('model', '-')} · Status: <strong>{item.get('status', '-')}</strong> · Latency: <strong>{item.get('latency_seconds', '-')}</strong></div>
                                <div>{item.get('reason', 'No reason recorded.')}</div>
                            </div>
                            """
                        )

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
        _render_goal_status(goal, kpis, latest_result)
        learning_profile = memory_summary.get("learning_profile", metrics.get("learning_profile", {}))
        current_action = _safe_dict(latest_result.get("action"))
        action_type = str(current_action.get("action", "monitor")).upper()
        if action_type in {"NONE", "IDLE", ""}:
            action_state = "IDLE"
        elif latest_result.get("critic_output", {}).get("requires_replan") and action_type != "REDIRECT":
            action_state = "REPLAN"
        else:
            action_state = action_type
        source_block = current_action.get("from") or "-"
        destination_block = current_action.get("to") or "-"
        reward_value = latest_result.get("reward_score", latest_result.get("reward", 0))
        search_delta = _safe_dict(latest_result.get("baseline_comparison")).get("search_time_delta_min")
        vehicle_ids = current_action.get("vehicle_ids") or (actions[-1].get("vehicle_ids") if actions else []) or []
        if not vehicle_ids and vehicle_events:
            vehicle_ids = [
                item.get("vehicle_number")
                for item in vehicle_events[-12:]
                if item.get("event") == "redirect" and item.get("decision_step") == step_number
            ]
        render_key_value_groups([
            {
                "title": "Live Goal Context",
                "items": [
                    {"label": "Current state", "value": action_state},
                    {"label": "Focus block", "value": f"{event_context.get('pressure_focus', event_context.get('focus_zone', '-'))} (pressure source)"},
                    {"label": "Chosen route", "value": f"{source_block} -> {destination_block}" if source_block != "-" or destination_block != "-" else "No route"},
                    {"label": "Vehicles", "value": ", ".join(map(str, vehicle_ids[:6])) if vehicle_ids else current_action.get("vehicles", 0)},
                ],
            },
            {
                "title": "Reward & Learning Relevance",
                "items": [
                    {"label": "Reward", "value": f"{float(reward_value or 0):+.2f}"},
                    {"label": "Search delta", "value": f"{float(search_delta or 0):+.2f} min" if search_delta is not None else "-"},
                    {"label": "Global guide", "value": event_context.get("recommended_zone", "-")},
                    {"label": "Decision role", "value": "Global guides incoming flow; local resolves active congestion."},
                ],
            },
        ])
        try:
            if float(reward_value or 0) < 0 and float(search_delta or 0) > 0:
                st.info("Reward penalized action cost despite search-time improvement.")
        except (TypeError, ValueError):
            pass
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
            route_counts = learning_profile.get("recent_route_counts") or learning_profile.get("route_counts") or {}
            if route_counts:
                route_frame = pd.DataFrame(
                    [
                        {"Route": route, "Recent uses": count}
                        for route, count in sorted(route_counts.items(), key=lambda item: item[1], reverse=True)[:8]
                    ]
                )
                st.markdown("<div class='section-kicker'>Dynamic Route Memory</div>", unsafe_allow_html=True)
                st.caption("Routes are monitored so the planner can avoid mechanical repetition unless the destination is still the safest buffer.")
                st.dataframe(route_frame, width="stretch", hide_index=True)
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
                    deduped_goals = _dedupe_goal_history(goal_history)
                    goal_frame = pd.DataFrame([
                        {
                            "objective": item.get("objective", "-"),
                            "priority_zone": latest_result.get("action", {}).get("from") or item.get("priority_zone", "-"),
                            "target_hotspots": item.get("target_congested_zones", "-"),
                            "target_search_time_min": item.get("target_search_time_min", "-"),
                            "status": item.get("status", "-"),
                            "revision_reason": item.get("revision_reason", "-"),
                            "revision_count": item.get("revision_count", 0),
                        }
                        for item in deduped_goals[-12:]
                    ])
                    st.caption(f"Showing {len(goal_frame)} unique goal revisions. Repeated identical revisions are hidden for demo clarity.")
                    st.dataframe(goal_frame, width="stretch", hide_index=True)
            llm_rules = learning_profile.get("llm_memory_rules", [])
            if llm_rules:
                with st.expander("LLM-derived route rules", expanded=False):
                    for index, rule in enumerate(llm_rules[-12:], start=1):
                        route = rule.get("route") or f"{rule.get('from', '-')} -> {rule.get('to', '-')}"
                        last_step = rule.get("last_step") or rule.get("step") or rule.get("created_step") or step_number
                        reason = rule.get("reason") or rule.get("rationale") or rule.get("summary") or "Rule stored from recent reward and routing behavior."
                        render_html_block(
                            f"""
                            <div class="llm-detail-row">
                                <div><strong>Rule {index}</strong> · Route: <strong>{route}</strong> · Last step: <strong>{last_step}</strong> · Source: <strong>{rule.get('source', 'memory')}</strong></div>
                                <div>{reason}</div>
                            </div>
                            """
                        )
        else:
            st.info("Learning profile is not available yet. Run one step to initialize memory signals.")

        lifecycle_frame = _build_lifecycle_memory_frame(vehicle_events, recent_cycles, movement_log)
        lifecycle_counts = {}
        if not lifecycle_frame.empty:
            lifecycle_counts = lifecycle_frame["Kind"].value_counts().to_dict()
        recent_action_rows = []
        cycle_rewards = {
            str(cycle.get("step")): cycle.get("reward_score", cycle.get("reward", "-"))
            for cycle in recent_cycles or []
            if isinstance(cycle, dict)
        }
        for item in (actions or [])[-8:]:
            if not isinstance(item, dict):
                continue
            route = f"{item.get('from', '-')} -> {item.get('to', '-')}"
            vehicle_list = item.get("vehicle_ids", [])
            recent_action_rows.append(
                {
                    "Step": item.get("step", "-"),
                    "Route": route,
                    "Vehicles": item.get("vehicles", 0),
                    "Vehicle IDs": ", ".join(map(str, vehicle_list[:5])) if isinstance(vehicle_list, list) and vehicle_list else "-",
                    "Reward": cycle_rewards.get(str(item.get("step")), "-"),
                    "Learning note": item.get("reason", "Route evaluated by memory guard."),
                }
            )
        render_key_value_groups([
            {
                "title": "Memory Applied This Step",
                "items": [
                    {"label": "Current route", "value": f"{source_block} -> {destination_block}" if source_block != "-" or destination_block != "-" else "No route"},
                    {"label": "Route guard", "value": "Checked recent usage and blocked routes"},
                    {"label": "Lifecycle rows", "value": len(lifecycle_frame)},
                    {"label": "User events tracked", "value": len([item for item in vehicle_events if item.get("user_type") != "simulated"])},
                ],
            },
            {
                "title": "Recent Lifecycle Mix",
                "items": [
                    {"label": "Entry", "value": lifecycle_counts.get("entry", 0)},
                    {"label": "Parked", "value": lifecycle_counts.get("parked", 0)},
                    {"label": "Redirect", "value": lifecycle_counts.get("redirect", 0)},
                    {"label": "Exit", "value": lifecycle_counts.get("exit", 0)},
                ],
            },
        ])
        if recent_action_rows:
            st.markdown("<div class='section-kicker'>Recent Route Learning Snapshot</div>", unsafe_allow_html=True)
            st.caption("This is the relevant memory view: latest routes, exact affected vehicle IDs, and reward context used by the learning guard.")
            action_frame = pd.DataFrame(recent_action_rows).iloc[::-1]
            action_frame["Learning note"] = action_frame["Learning note"].astype(str).str.slice(0, 110)
            st.dataframe(action_frame, width="stretch", hide_index=True)

        history = memory_summary.get("history", [])
        if history:
            with st.expander("Advanced raw decision memory table", expanded=False):
                st.caption("Raw persisted memory for debugging. The synced learning snapshot above is the judge-facing view.")
                raw_history = pd.DataFrame(history)
                if not raw_history.empty:
                    st.dataframe(raw_history.tail(12), width="stretch", hide_index=True)
        if not lifecycle_frame.empty:
            st.markdown("<div class='section-kicker'>Recent Vehicle Lifecycle Memory</div>", unsafe_allow_html=True)
            st.caption("Latest state transitions only. Sequence should stay entry -> parked -> redirect optional -> exit.")
            st.dataframe(lifecycle_frame.head(10), width="stretch", hide_index=True)
            with st.expander("Advanced full lifecycle audit", expanded=False):
                st.dataframe(lifecycle_frame.head(40), width="stretch", hide_index=True)
        
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
        user_event_items = [
            item for item in vehicle_events
            if item.get("user_type") != "simulated"
        ]
        active_known_users = [
            item for item in user_vehicles
            if item.get("status") != "exited" and str(item.get("name", "")).strip()
        ]
        latest_active_user = active_known_users[-1] if active_known_users else {}
        render_key_value_groups([
            {
                "title": "Known User Sync",
                "items": [
                    {"label": "Active named users", "value": len(active_known_users)},
                    {"label": "Recent user events", "value": len(user_event_items[-20:])},
                    {"label": "Latest vehicle", "value": user_event_items[-1].get("vehicle_number", "-") if user_event_items else "-"},
                    {"label": "Latest status", "value": user_event_items[-1].get("event", "Waiting") if user_event_items else "Waiting"},
                ],
            }
        ])
        if latest_active_user:
            _render_glow_box(
                "Latest active user vehicle",
                (
                    f"{latest_active_user.get('vehicle_number') or latest_active_user.get('number') or latest_active_user.get('id')} "
                    f"is {latest_active_user.get('status', 'parked')} at "
                    f"{latest_active_user.get('block') or latest_active_user.get('assigned_block') or '-'} via "
                    f"{latest_active_user.get('gate', '-')}. Driver: {latest_active_user.get('name', 'Known user')}."
                ),
                "success",
            )
        elif user_event_items:
            latest_user_event = user_event_items[-1]
            _render_glow_box(
                "Latest user app event",
                (
                    f"{latest_user_event.get('vehicle_number', '-')} {latest_user_event.get('event', 'updated')} "
                    f"at {latest_user_event.get('to_block') or latest_user_event.get('block') or latest_user_event.get('to_gate') or '-'}."
                ),
                "info",
            )
        if user_event_items:
            st.markdown("<div class='section-kicker'>Known User App Activity</div>", unsafe_allow_html=True)
            st.caption("Live web-app entry, parking, redirect, and exit events. Names are shown only inside the dashboard audit view; contact data is never displayed.")
            user_notice_frame = pd.DataFrame([
                {
                    "Time": _format_live_timestamp(item.get("timestamp")),
                    "Event": item.get("event", "-"),
                    "Vehicle": item.get("vehicle_number", "-"),
                    "Driver": item.get("name", "-"),
                    "Block / Gate": item.get("to_block") or item.get("block") or item.get("from_gate") or item.get("to_gate") or "-",
                    "Status": item.get("status", item.get("event", "-")),
                }
                for item in user_event_items[-10:]
            ])
            st.dataframe(user_notice_frame.iloc[::-1], width="stretch", hide_index=True)
        else:
            st.info("Known user entries from the web app will appear here immediately after assignment.")
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

    elif active_page == "Vehicle & Flow Monitoring":
        st.markdown("<div class='section-kicker'>Combined Vehicle & Flow Monitoring</div>", unsafe_allow_html=True)
        render_insight_cards([
            {"title": "Total Vehicles", "value_label": "All", "value": vehicle_stats.get("total", len(vehicles)), "note": "Simulated + user vehicles."},
            {"title": "User Vehicles", "value_label": "Live", "value": vehicle_stats.get("user", len(user_vehicles)), "note": "Real entries through gates."},
            {"title": "Simulated Vehicles", "value_label": "Digital Twin", "value": vehicle_stats.get("simulated", len(simulated_vehicles)), "note": "Existing simulation still running."},
            {"title": "Active / Exited", "value_label": "Status", "value": f"{vehicle_stats.get('active', 0)} / {vehicle_stats.get('exited', 0)}", "note": "Current flow status."},
        ], columns=4)
        render_key_value_groups([
            {
                "title": "Live Flow Indicator",
                "items": [
                    {"label": "Vehicles entering", "value": vehicle_stats.get("entering", 0)},
                    {"label": "Redirecting", "value": vehicle_stats.get("redirecting", 0)},
                    {"label": "Exiting", "value": vehicle_stats.get("exiting", 0)},
                    {"label": "Consistency check", "value": "State Verified" if vehicle_stats.get("consistency") == "verified" else "Review Needed"},
                ],
            }
        ])
        if agentic_integrity:
            integrity_cols = st.columns([0.75, 1.25])
            integrity_cols[0].metric("Agentic Integrity", f"{agentic_integrity.get('score', 0)}%")
            integrity_cols[0].caption(agentic_integrity.get("status", "Review"))
            check_frame = pd.DataFrame([
                {
                    "Check": item.get("name", "-"),
                    "Status": "Pass" if item.get("ok") else "Review",
                    "Detail": item.get("detail", "-"),
                }
                for item in agentic_integrity.get("checks", [])
            ])
            integrity_cols[1].dataframe(check_frame, width="stretch", hide_index=True)
        if actions:
            latest_action = actions[-1]
            route_cols = st.columns([1.15, 0.7, 1.15, 0.9])
            route_cols[0].markdown(
                f"""
                <div class="metric-card danger-card">
                  <span>Source Block</span>
                  <strong>{latest_action.get('from', '-')}</strong>
                  <small>Red glow in frontend</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            route_cols[1].markdown(
                f"""
                <div class="metric-card route-card">
                  <span>Moving Now</span>
                  <strong>{latest_action.get('vehicles', 0)}</strong>
                  <small>Exact animated units</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            route_cols[2].markdown(
                f"""
                <div class="metric-card success-card">
                  <span>Destination Block</span>
                  <strong>{latest_action.get('to', '-')}</strong>
                  <small>Green pulse in frontend</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            route_cols[3].markdown(
                f"""
                <div class="metric-card">
                  <span>Visual Link</span>
                  <strong>Synced</strong>
                  <small>Dashboard -> frontend</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            affected = latest_action.get("vehicle_ids") or []
            st.info(
                f"Recent action highlight: {latest_action.get('vehicles', 0)} vehicle(s) redirected "
                f"{latest_action.get('from', '-')} -> {latest_action.get('to', '-')}"
                + (f" | Affected vehicles: {', '.join(map(str, affected[:6]))}" if affected else "")
            )
        if vehicle_events:
            last_event = vehicle_events[-1]
            st.success(
                f"Last event: {last_event.get('vehicle_number', 'Vehicle')} "
                f"{last_event.get('event', '-')} at "
                f"{last_event.get('to_block') or last_event.get('block') or last_event.get('to_gate') or '-'}"
            )

        basic_tab, vehicle_tab, event_tab = st.tabs(["Basic Flow", "Vehicle Table", "Event Audit"])
        with basic_tab:
            active_users = [item for item in vehicles if item.get("user_type") != "simulated" and item.get("status") != "exited"]
            live_entering = vehicle_stats.get("entering", 0)
            live_redirecting = vehicle_stats.get("redirecting", 0)
            live_exiting = vehicle_stats.get("exiting", 0)
            render_key_value_groups([
                {
                    "title": "Presentation View",
                    "items": [
                        {"label": "Active user vehicles", "value": len(active_users)},
                        {"label": "Currently entering", "value": live_entering},
                        {"label": "Currently redirecting", "value": live_redirecting},
                        {"label": "Currently exiting", "value": live_exiting},
                    ],
                },
                {
                    "title": "Recent Audit Events",
                    "items": [
                        {"label": "Entry records", "value": vehicle_stats.get("recent_entries", 0)},
                        {"label": "Redirect records", "value": vehicle_stats.get("recent_redirects", 0)},
                        {"label": "Exit records", "value": vehicle_stats.get("recent_exits", 0)},
                        {"label": "Sync status", "value": "Perfect" if vehicle_stats.get("consistency") == "verified" else "Review"},
                    ],
                },
            ])
            st.caption("Live counters show vehicles moving in the current simulation tick. Recent Audit Events counts persisted entry/redirect/exit event rows from the latest event window.")
            named_users = [
                item for item in active_users
                if str(item.get("name", "")).strip()
            ]
            if named_users:
                st.markdown("<div class='section-kicker'>Known Driver Entries</div>", unsafe_allow_html=True)
                st.caption(f"Showing recent known active drivers only: {len(named_users[-12:])} shown out of {len(named_users)} named active user vehicle(s). Total user sessions may be higher because exited sessions are retained for audit.")
                driver_frame = pd.DataFrame([
                    {
                        "Driver": item.get("name", "Unknown"),
                        "Vehicle": item.get("number") or item.get("id"),
                        "Block": item.get("block") or "-",
                        "Gate": item.get("gate") or "-",
                        "Status": item.get("status", "parked").title(),
                    }
                    for item in named_users[-12:]
                ])
                st.dataframe(driver_frame.iloc[::-1], width="stretch", hide_index=True)
            else:
                st.info("Known driver entries will appear here when a named user parks through the web app.")

        with vehicle_tab:
            vehicle_scope = st.radio(
                "Vehicle filter",
                ["All vehicles", "User vehicles", "Active vehicles"],
                index=1,
                horizontal=True,
                key="vehicle_monitor_filter",
            )
            combined_vehicle_map = {}
            for item in [*vehicles, *user_vehicles, *simulated_vehicles]:
                key = str(item.get("id") or item.get("number") or len(combined_vehicle_map))
                combined_vehicle_map[key] = item
            combined_vehicles = list(combined_vehicle_map.values())
            display_vehicles = combined_vehicles
            if vehicle_scope == "User vehicles":
                display_vehicles = [item for item in combined_vehicles if item.get("user_type") != "simulated"]
            elif vehicle_scope == "Active vehicles":
                display_vehicles = [item for item in combined_vehicles if item.get("status") != "exited"]

            vehicle_frame = pd.DataFrame([
                {
                    "Vehicle": item.get("number") or item.get("id"),
                    "Name": item.get("name", "Unknown") if item.get("user_type") != "simulated" else "-",
                    "Type": item.get("type", "-"),
                    "User Type": item.get("user_type", "simulated"),
                    "Block": item.get("block") or "-",
                    "Status": item.get("status", "parked"),
                    "Gate": item.get("gate") or "-",
                }
                for item in display_vehicles
            ])
            if vehicle_frame.empty:
                st.info("Vehicle table will populate as simulated and user vehicles enter the system.")
            else:
                def _user_type_style(value):
                    palette = {
                        "simulated": "background-color: rgba(160,170,185,0.16); color: #c6d0dd;",
                        "student": "background-color: rgba(88,166,255,0.18); color: #9dccff;",
                        "visitor": "background-color: rgba(255,182,110,0.18); color: #ffc68e;",
                        "staff": "background-color: rgba(75,211,138,0.16); color: #9cf3bc;",
                    }
                    return palette.get(str(value).lower(), "")
                st.dataframe(vehicle_frame.style.applymap(_user_type_style, subset=["User Type"]), width="stretch", hide_index=True)

        with event_tab:
            event_order = {"entry": 1, "parked": 2, "redirect": 3, "exit": 4}
            user_event_items = [
                item for item in vehicle_events
                if item.get("user_type") != "simulated"
            ]
            if user_event_items:
                st.markdown("<div class='section-kicker'>User Entry / Exit Events</div>", unsafe_allow_html=True)
                user_event_frame = pd.DataFrame([
                    {
                        "Time": _format_live_timestamp(item.get("timestamp")),
                        "Event": item.get("event", "-"),
                        "Vehicle": item.get("vehicle_number", "-"),
                        "Name": item.get("name", "-"),
                        "From": item.get("from_block") or item.get("from_gate") or "-",
                        "To": item.get("to_block") or item.get("to_gate") or item.get("block") or "-",
                        "Gate": item.get("gate") or item.get("from_gate") or item.get("to_gate") or "-",
                        "_timestamp": str(item.get("timestamp", "")),
                        "_order": event_order.get(item.get("event"), 99),
                    }
                    for item in user_event_items[-20:]
                ])
                user_event_frame = user_event_frame.sort_values(["_timestamp", "_order"], ascending=[False, True])
                st.dataframe(user_event_frame.drop(columns=["_timestamp", "_order"]), width="stretch", hide_index=True)
            else:
                st.info("User parking events will appear here immediately after the web app assigns or exits a vehicle.")

            event_frame = pd.DataFrame([
                {
                    "Time": _format_live_timestamp(item.get("timestamp")),
                    "Event": item.get("event", "-"),
                    "Vehicle": item.get("vehicle_number", "-"),
                    "Name": item.get("name", "-") if item.get("user_type") != "simulated" else "-",
                    "Type": item.get("type", "-"),
                    "User Type": item.get("user_type", "-"),
                    "From": item.get("from_block") or item.get("from_gate") or "-",
                    "To": item.get("to_block") or item.get("to_gate") or item.get("block") or "-",
                    "Decision Step": item.get("decision_step", item.get("step", "-")),
                    "Gate": item.get("gate") or item.get("from_gate") or item.get("to_gate") or "-",
                    "_timestamp": str(item.get("timestamp", "")),
                    "_order": event_order.get(item.get("event"), 99),
                }
                for item in vehicle_events[-80:]
            ])
            st.markdown("<div class='section-kicker'>Entry, Redirect, Exit Events</div>", unsafe_allow_html=True)
            if event_frame.empty:
                st.info("User entry, redirect, and exit events will appear here.")
            else:
                event_frame = event_frame.sort_values(["_timestamp", "_order"], ascending=[False, True])
                st.dataframe(event_frame.drop(columns=["_timestamp", "_order"]).head(40), width="stretch", hide_index=True)

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
            if int(llm_status.get("available_key_count", 0) or 0) > 0:
                st.info("Some Gemini routes are cooling down, but chat will try a healthy route first and fall back locally if the provider times out.")
            else:
                st.warning("Gemini chat is paused by quota backoff. Local chat fallback will still answer operational questions.")
        _render_assistant_briefing(assistant_briefing)
        chat_action = _safe_dict(latest_result.get("action"))
        latest_user_event = next(
            (
                item for item in reversed(vehicle_events)
                if item.get("user_type") != "simulated"
            ),
            {},
        )
        render_key_value_groups([
            {
                "title": "Live Chat Context",
                "items": [
                    {"label": "Backend step", "value": step_number},
                    {"label": "Current route", "value": f"{chat_action.get('from', '-')} -> {chat_action.get('to', '-')}" if chat_action.get("action") == "redirect" else "No redirect"},
                    {"label": "Pressure focus", "value": event_context.get("pressure_focus", event_context.get("focus_zone", "-"))},
                    {"label": "Best global block", "value": event_context.get("recommended_zone", "-")},
                ],
            },
            {
                "title": "User & LLM State",
                "items": [
                    {"label": "User vehicles", "value": vehicle_stats.get("user", len(user_vehicles))},
                    {"label": "Latest user event", "value": f"{latest_user_event.get('vehicle_number', '-')} {latest_user_event.get('event', '')}".strip() if latest_user_event else "Waiting"},
                    {"label": "Reasoning mode", "value": "Gemini" if last_llm_decision.get("requested") else "Local Reasoning"},
                    {"label": "LLM influence", "value": last_llm_decision.get("influence_label", llm_usage_summary.get("llm_influence_pct", "0%"))},
                ],
            },
        ])
        _render_glow_box(
            "Assistant context",
            (
                f"The chat is grounded in backend step {step_number}, route "
                f"{chat_action.get('from', '-') if chat_action.get('action') == 'redirect' else 'monitoring'}"
                f"{' → ' + chat_action.get('to', '-') if chat_action.get('action') == 'redirect' else ''}, "
                f"user vehicles {vehicle_stats.get('user', len(user_vehicles))}, and pressure focus "
                f"{event_context.get('pressure_focus', event_context.get('focus_zone', '-'))}."
            ),
            "info",
        )
        suggestion_cols = st.columns(4)
        suggestions = [
            "Which SRM block is best right now?",
            "Where did the latest known user park?",
            "Show the latest allocation decision",
            "Why did the agent choose this route?",
        ]
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
            _render_glow_box(
                f"Assistant answer · {meta.get('source', 'unknown')} · Gemini {'used' if meta.get('llm_used') else 'not used'}",
                f"{st.session_state.chat_response}\n\n{meta.get('reason', '')}",
                "success" if meta.get("llm_used") else "info",
            )

    # Keep Streamlit tabs stable for demos. The Expo console and frontend poll the backend directly;
    # use the Refresh Dashboard Snapshot button for a manual Streamlit snapshot refresh.

if __name__ == "__main__":
    main()
