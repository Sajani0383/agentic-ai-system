const API_BASE = "http://127.0.0.1:8000";
const EMPTY_STATE = {
  step: 0,
  updated_at: "",
  blocks: {},
  vehicles: [],
  simulated_vehicles: [],
  user_vehicles: [],
  users: [],
  events: [],
  gates: {},
  vehicle_stats: {},
  actions: [],
  movement_log: [],
  alerts: [],
  latest_decision: {},
  latest_result: {},
  event_context: {},
  reasoning_summary: {},
  llm_usage_summary: {},
  decision_reason: "",
  agent_thought: "",
  learning: {},
  llm: {},
};

const app = document.getElementById("app");
let latestState = { ...EMPTY_STATE };
let previousState = { ...EMPTY_STATE };
let selectedBlock = null;
let selectedVehicleId = null;
let lastAnimatedStep = -1;
let animatedEventIds = new Set();
let lastRenderKey = "";
let viewMode = "basic";
try {
  viewMode = localStorage.getItem("parkingViewMode") || "basic";
} catch (error) {
  viewMode = "basic";
}
let showAllVehicles = false;
let timelineExpanded = false;

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN").format(Number(value || 0));
}

function formatTimestamp(value) {
  if (!value) return "Waiting for backend";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function shortText(value, limit = 180) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 3).trim()}...`;
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function occupancyLevel(occupied, capacity) {
  const ratio = capacity ? occupied / capacity : 0;
  if (ratio >= 0.7) return "high";
  if (ratio >= 0.4) return "medium";
  return "low";
}

function campusPosition(name, index, total) {
  const predefined = {
    "Main Block": [8, 12],
    "Hi-Tech Block": [36, 10],
    "ES Block": [62, 13],
    "Mech A": [14, 39],
    "Mech B": [34, 42],
    "Mech C": [55, 39],
    "Automobile Block": [76, 42],
    "CRC Block": [19, 66],
    "Admin Block": [44, 64],
    "Library": [66, 65],
    "MBA Block": [83, 67],
    "Biotech Block": [13, 86],
    "Tech Park": [48, 86],
    "Basic Eng Lab": [77, 86],
  };
  if (predefined[name]) return { left: predefined[name][0], top: predefined[name][1] };
  const angle = (index / Math.max(1, total)) * Math.PI * 2;
  return { left: 48 + Math.cos(angle) * 38, top: 50 + Math.sin(angle) * 36 };
}

function routeAngle(from, to) {
  const dx = Number(to.left || 0) - Number(from.left || 0);
  const dy = Number(to.top || 0) - Number(from.top || 0);
  return Math.atan2(dy, dx) * 180 / Math.PI;
}

function realisticVehicleMarkup(type = "car", label = "", extraClass = "") {
  const vehicleType = type === "bike" ? "bike" : "car";
  const safeLabel = label ? ` title="${String(label).replace(/"/g, "&quot;")}"` : "";
  return `
    <span class="vehicle-sprite ${vehicleType} ${extraClass}"${safeLabel} aria-hidden="true">
      <span class="vehicle-body"></span>
      <span class="vehicle-window"></span>
      <span class="vehicle-wheel front"></span>
      <span class="vehicle-wheel rear"></span>
    </span>
  `;
}

function buildRoadNetwork() {
  return `
    <div class="map-grid"></div>
    <svg class="road-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <path class="road-shadow" d="M-4 50 C14 45, 27 46, 42 51 S72 61, 104 46"></path>
      <path class="road-main" d="M-4 50 C14 45, 27 46, 42 51 S72 61, 104 46"></path>
      <path class="road-dash" d="M-4 50 C14 45, 27 46, 42 51 S72 61, 104 46"></path>
      <path class="road-shadow" d="M50 -4 C47 18, 53 31, 49 47 S44 74, 51 104"></path>
      <path class="road-main" d="M50 -4 C47 18, 53 31, 49 47 S44 74, 51 104"></path>
      <path class="road-dash" d="M50 -4 C47 18, 53 31, 49 47 S44 74, 51 104"></path>
      <path class="road-secondary" d="M8 28 C24 35, 33 39, 48 50"></path>
      <path class="road-secondary" d="M89 29 C73 37, 62 42, 50 51"></path>
      <path class="road-secondary" d="M11 77 C25 70, 36 64, 49 52"></path>
      <path class="road-secondary" d="M90 75 C76 69, 63 62, 50 52"></path>
    </svg>
    <div class="campus-green green-a"></div>
    <div class="campus-green green-b"></div>
    <div class="campus-water"></div>
    <div class="campus-building admin-core"></div>
    <div class="campus-building academic-core"></div>
    <div class="campus-parking-lot lot-a"></div>
    <div class="campus-parking-lot lot-b"></div>
    <div class="campus-parking-lot lot-c"></div>
    <div class="campus-label north">Academic Avenue</div>
    <div class="campus-label south">Research Road</div>
    <div class="compass">N</div>
    <div class="map-intersection center"></div>
    <div class="map-intersection south"></div>
  `;
}

function buildMapMotion(state, entries) {
  const action = activeRouteAction(state);
  const stats = state.vehicle_stats || {};
  const gates = state.gates || {};
  if (action?.type === "redirect") {
    const sourceIndex = entries.findIndex(([name]) => name === action.from);
    const destIndex = entries.findIndex(([name]) => name === action.to);
    if (sourceIndex < 0 || destIndex < 0) return "";
    const source = campusPosition(action.from, sourceIndex, entries.length);
    const dest = campusPosition(action.to, destIndex, entries.length);
    const count = Math.max(1, Math.min(3, Number(action.vehicles || 1)));
    return Array.from({ length: count }, (_, index) => `
      <div class="map-moving-vehicle redirect unit-${index}" style="--x1:${source.left}%; --y1:${source.top}%; --x2:${dest.left}%; --y2:${dest.top}%; --angle:${routeAngle(source, dest)}deg; --lane:${(index - (count - 1) / 2) * 3}px; --delay:${index * 0.22}s;">
        ${realisticVehicleMarkup(index < Number(action.car_vehicles || count) ? "car" : "bike", "", "moving")}
      </div>
    `).join("");
  }
  const entryCount = Math.max(0, Math.min(2, Number(stats.entering || 0)));
  if (!entryCount || !entries.length) return "";
  const targetIndex = entries.findIndex(([name]) => name === ((state.events || []).slice(-1)[0]?.to_block));
  const target = campusPosition(targetIndex >= 0 ? entries[targetIndex][0] : entries[0][0], Math.max(0, targetIndex), entries.length);
  const gateList = Object.entries(gates);
  if (!gateList.length) return "";
  return Array.from({ length: entryCount }, (_, index) => {
    const [, gate] = gateList[index % gateList.length];
    const source = { left: Number(gate.x || 0.5) * 100, top: Number(gate.y || 0.5) * 100 };
    return `
      <div class="map-moving-vehicle entry unit-${index}" style="--x1:${source.left}%; --y1:${source.top}%; --x2:${target.left}%; --y2:${target.top}%; --angle:${routeAngle(source, target)}deg; --lane:${(index - (entryCount - 1) / 2) * 3}px; --delay:${index * 0.2}s;">
        ${realisticVehicleMarkup("car", "", "moving")}
      </div>
    `;
  }).join("");
}

function buildMiniMap(state) {
  const entries = Object.entries(state.blocks || {});
  const latestAction = activeRouteAction(state);
  const gateNodes = Object.entries(state.gates || {}).map(([name, gate]) => `
    <div class="map-gate" data-gate="${name}" style="left:${Number(gate.x || 0.5) * 100}%; top:${Number(gate.y || 0.5) * 100}%;" title="${name}">
      <span>${name}</span>
    </div>
  `).join("");
  const nodes = entries.map(([name, block], index) => {
    const occupied = Number(block.occupied || 0);
    const capacity = Number(block.capacity || 0);
    const free = Math.max(0, capacity - occupied);
    const pos = campusPosition(name, index, entries.length);
    const level = occupancyLevel(occupied, capacity);
    const source = latestAction?.from === name ? " source-node" : "";
    const destination = latestAction?.to === name ? " destination-node" : "";
    return `
      <button class="map-node ${level}${source}${destination}" data-block="${name}" style="left:${pos.left}%; top:${pos.top}%;" title="${name}: ${occupied}/${capacity} occupied">
        <span class="map-node-name">${name}</span>
        <span class="map-node-count">${formatNumber(occupied)} / ${formatNumber(capacity)}</span>
        <span class="map-node-free">${formatNumber(free)} free</span>
      </button>
    `;
  }).join("");
  let arrow = "";
  if (latestAction?.type === "redirect") {
    const sourceIndex = entries.findIndex(([name]) => name === latestAction.from);
    const destIndex = entries.findIndex(([name]) => name === latestAction.to);
    if (sourceIndex >= 0 && destIndex >= 0) {
      const source = campusPosition(latestAction.from, sourceIndex, entries.length);
      const dest = campusPosition(latestAction.to, destIndex, entries.length);
      arrow = `
        <svg class="map-arrow" viewBox="0 0 100 100" preserveAspectRatio="none">
          <line x1="${source.left}" y1="${source.top}" x2="${dest.left}" y2="${dest.top}"></line>
        </svg>
      `;
    }
  }
  return `
    <section class="map-panel">
      <div class="board-header">
        <div>
          <h2>Live Campus Road Map</h2>
          <span>Google-map style road view with gates, congestion heat, and backend-synced vehicle motion.</span>
        </div>
        <div class="map-legend">
          <span><i class="low"></i>Open</span>
          <span><i class="medium"></i>Busy</span>
          <span><i class="high"></i>Full</span>
        </div>
      </div>
      <div class="campus-map">${buildRoadNetwork()}${arrow}${buildMapMotion(state, entries)}${gateNodes}${nodes}</div>
    </section>
  `;
}

function buildZoneSlotOverview(state) {
  const entries = Object.entries(state.blocks || {});
  if (!entries.length) return "";
  const action = activeRouteAction(state);
  return `
    <section class="zone-overview-panel">
      <div class="board-header">
        <div>
          <h2>Zone Slot Overview</h2>
          <span>Parked slots by zone. Click any zone for the full slot-level layout.</span>
        </div>
      </div>
      <div class="zone-overview-grid">
        ${entries.map(([name, block]) => {
          const capacity = Math.max(0, Number(block.capacity || 0));
          const occupied = Math.min(capacity, Math.max(0, Number(block.occupied || 0)));
          const free = Math.max(0, capacity - occupied);
          const level = occupancyLevel(occupied, capacity);
          const active = action && (action.from === name || action.to === name) ? " active" : "";
          const role = action?.from === name ? "Source" : action?.to === name ? "Destination" : "";
          return `
            <button class="zone-overview-card ${level}${active}" data-block="${escapeHtml(name)}">
              <strong>${escapeHtml(name)}</strong>
              <span>${formatNumber(occupied)} parked / ${formatNumber(capacity)} slots</span>
              <small>${formatNumber(free)} free · Car Slots ${formatNumber(block.car_slots || 0)} · Bike Slots ${formatNumber(block.bike_slots || 0)}${role ? ` · ${role}` : ""}</small>
              <div class="zone-bar"><i style="width:${capacity ? Math.round((occupied / capacity) * 100) : 0}%"></i></div>
            </button>
          `;
        }).join("")}
      </div>
    </section>
  `;
}

function buildDecisionPanel(state) {
  const action = state.latest_decision || {};
  const latestAction = state.actions?.length ? state.actions[state.actions.length - 1] : null;
  const actionName = String(action.action || "").toLowerCase();
  const actionState = actionName === "redirect" ? "REDIRECT" : actionName === "replan" ? "REPLAN" : "IDLE";
  const actionText = actionState;
  const route = action.action === "redirect"
    ? `${action.from || latestAction?.from || "-"} -> ${action.to || latestAction?.to || "-"}`
    : "No active route transfer";
  const eventContext = state.event_context || {};
  const globalBlock = eventContext.recommended_zone || "-";
  const focusBlock = eventContext.pressure_focus || eventContext.focus_zone || "-";
  const sourceBlock = action.from || latestAction?.from || "-";
  const perceptionLink = focusBlock !== "-" && sourceBlock !== "-" && focusBlock !== sourceBlock
    ? `${focusBlock} pressure triggered upstream rerouting via ${sourceBlock}.`
    : `Focus block ${focusBlock}; source block ${sourceBlock}.`;
  const affectedIds = affectedVehicleIds(state).slice(0, 5);
  const rewardScore = Number(state.latest_result?.reward_score ?? state.reasoning_summary?.reward_score ?? 0);
  const rewardNote = rewardScore < -0.05
    ? "Reward penalized action cost despite search-time improvement."
    : "Reward aligned with current route outcome.";
  const llm = state.llm || {};
  const changedFields = Array.isArray(llm.changed_fields) ? llm.changed_fields : [];
  const llmText = llm.used
    ? `LLM ${llm.influence_label || "Confirmed"}${changedFields.length ? `: changed ${changedFields.join(", ").replace("_", " ")}` : ""}`
    : (llm.requested ? "LLM requested; local execution used" : "Local Reasoning");
  const localAction = llm.local_action || {};
  const llmAction = llm.llm_action || {};
  const finalAction = llm.final_action || action;
  const llmComparison = llm.requested ? `
    <div class="llm-comparison">
      <div><span>Local</span><strong>${formatActionCompact(localAction)}</strong></div>
      <div><span>Gemini</span><strong>${formatActionCompact(llmAction)}</strong></div>
      <div><span>Final</span><strong>${formatActionCompact(finalAction)}</strong></div>
    </div>
  ` : "";
  return `
    <section class="decision-panel ${action.action === "redirect" ? "flash-decision" : ""}">
      <div>
        <div class="eyebrow">Current Decision</div>
        <h2>${actionText}</h2>
        <p>${route}</p>
        <div class="decision-ids">${affectedIds.length ? `Vehicles: ${affectedIds.join(", ")}` : "Vehicles: synced from backend"}</div>
      </div>
      <div class="decision-detail">
        <div class="llm-badge ${llm.influence === "modified" ? "modified" : "confirmed"}">${llm.used || llm.requested ? `LLM Influence: ${llm.influence_label || "Reviewed"}` : "Local Reasoning"}</div>
        <strong>${state.agent_thought || state.decision_reason || "Planner is monitoring live parking state."}</strong>
        <span>Recommended global block: ${globalBlock}. Chosen local route: ${route}. Global handles incoming flow; local resolves active congestion.</span>
        <span>Focus = pressure source. Source = rerouting origin. ${perceptionLink}</span>
        <span class="reward-note">${rewardNote}</span>
        <small>${shortText(llm.influence_summary || `${llmText}${llm.summary ? ` - ${llm.summary}` : ""}`, 210)}</small>
        ${llm.summary ? `<details class="llm-details"><summary>${llm.used || llm.requested ? "Gemini Reasoning" : "Local Reasoning"}</summary><small>${shortText(llm.summary, 520)}</small></details>` : ""}
        ${llmComparison}
      </div>
    </section>
  `;
}

function affectedVehicleIds(state) {
  const action = activeRouteAction(state);
  if (Array.isArray(action?.vehicle_ids) && action.vehicle_ids.filter(Boolean).length) {
    return action.vehicle_ids.filter(Boolean).map(String);
  }
  return (state.events || [])
    .filter((event) => event.event === "redirect" && String(event.decision_step || event.step || "") === String(state.step || ""))
    .slice(0, 6)
    .map((event) => event.vehicle_number || event.vehicle_id)
    .filter(Boolean)
    .map(String);
}

function activeRouteAction(state) {
  const latestAction = state.actions?.length ? state.actions[state.actions.length - 1] : null;
  const decision = state.latest_decision || {};
  if (decision.action === "redirect") {
    return {
      type: "redirect",
      from: decision.from,
      to: decision.to,
      vehicles: decision.vehicles,
      car_vehicles: decision.car_vehicles || decision.vehicles || 0,
      bike_vehicles: decision.bike_vehicles || 0,
      reason: decision.reason || state.decision_reason || "Agent route execution",
      step: state.step,
      timestamp: state.updated_at,
      vehicle_ids: decision.vehicle_ids || [],
    };
  }
  if (latestAction?.type === "redirect") return latestAction;
  return null;
}

function lastFlowEvent(state) {
  return (state.events || []).slice().reverse().find((event) => ["entry", "redirect", "exit"].includes(event.event));
}

function buildLiveRouteStage(state) {
  const action = activeRouteAction(state);
  const event = lastFlowEvent(state);
  const isRedirect = action?.type === "redirect";
  const source = isRedirect ? action.from : (event?.from_gate || event?.gate || "Gate1");
  const destination = isRedirect ? action.to : (event?.to_block || event?.block || event?.to_gate || "Waiting");
  const count = Math.max(1, Math.min(3, Number(isRedirect ? action.vehicles : 1) || 1));
  const affected = Array.isArray(action?.vehicle_ids) ? action.vehicle_ids.filter(Boolean).slice(0, 4) : [];
  const mode = isRedirect ? "Redirect In Motion" : event ? `${String(event.event || "Flow").toUpperCase()} Flow` : "Awaiting Movement";
  const reason = isRedirect ? action.reason : event ? `${event.vehicle_number || "Vehicle"} ${event.event}` : "Next backend route will appear here.";
  return `
    <section class="route-stage ${isRedirect ? "active" : ""}">
      <div class="route-stage-head">
        <div>
          <div class="eyebrow">${mode}</div>
          <h2>${source || "-"} <span>to</span> ${destination || "-"}</h2>
        </div>
        <div class="route-count">
          <span>Visible Units</span>
          <strong>${isRedirect ? formatNumber(action.vehicles || 0) : event ? "1" : "0"}</strong>
        </div>
      </div>
      <div class="route-lane" aria-label="Live vehicle movement">
        <div class="route-end source"><span>${source || "-"}</span><small>${isRedirect ? "Source block" : "Gate"}</small></div>
        <div class="route-track">
          ${Array.from({ length: count }, (_, index) => `
            <span class="route-car unit-${index}" style="--delay:${index * 0.18}s">
              ${realisticVehicleMarkup(index < Number(action?.car_vehicles || count) ? "car" : "bike", "", "moving")}
            </span>
          `).join("")}
        </div>
        <div class="route-end destination"><span>${destination || "-"}</span><small>${isRedirect ? "Destination block" : "Target"}</small></div>
      </div>
      <div class="route-stage-foot">
        <span>${shortText(reason, 130)}</span>
        ${affected.length ? `<strong>Affected: ${affected.join(", ")}</strong>` : "<strong>Exact vehicle count mirrored from backend</strong>"}
      </div>
    </section>
  `;
}

function formatActionCompact(action) {
  if (!action || action.action !== "redirect") return "Monitor";
  return `${action.from || "-"} -> ${action.to || "-"} (${action.vehicles || 0})`;
}

function buildLLMRouterPanel(state) {
  const llm = state.llm || {};
  const trace = Array.isArray(llm.router_trace) ? llm.router_trace.slice(-6).reverse() : [];
  if (!llm.router_mode && !trace.length) return "";
  return `
    <section class="router-panel">
      <div class="board-header">
        <div>
          <h2>LLM Orchestrator</h2>
          <span>${llm.router_mode || "Single-Key Gemini"}${llm.active_route?.model ? ` · Active model ${llm.active_route.model}` : ""}</span>
        </div>
      </div>
      ${trace.length ? `<div class="router-steps">${trace.map((item) => `
        <div class="router-step ${item.status || "pending"}">
          <strong>${item.key || "-"}</strong>
          <span>${item.model || "-"}</span>
          <small>${item.status || "pending"}${item.reason ? ` · ${item.reason}` : ""}</small>
        </div>
      `).join("")}</div>` : `<div class="fallback-panel compact"><span>Router trace will appear after the next Gemini attempt.</span></div>`}
    </section>
  `;
}

function buildAlertsPanel(state) {
  const alerts = state.alerts || [];
  return `
    <section class="alert-panel">
      <div class="board-header">
        <div>
          <h2>Parking App Notifications</h2>
          <span>Alerts shown here represent messages that car and bike users would receive in the SRM parking app.</span>
        </div>
      </div>
      ${alerts.length ? `<div class="alert-list">${alerts.slice(0, 5).map((alert) => `
        <div class="alert-item ${alert.level || "info"}">
          <strong>${alert.title || "SRM parking update"}</strong>
          <span>${alert.message || ""}</span>
          <small>${alert.audience || "parking_app_users"}</small>
        </div>
      `).join("")}</div>` : `<div class="fallback-panel compact"><span>No active alerts. Parking app users remain on normal routing.</span></div>`}
    </section>
  `;
}

function buildNotificationStrip(state) {
  const alerts = state.alerts || [];
  const items = alerts.slice(0, 3);
  return `
    <section class="alert-panel compact-alerts">
      <div class="board-header">
        <div>
          <h2>Live Notifications</h2>
          <span>Only the latest user-facing system messages are shown in Basic View.</span>
        </div>
      </div>
      ${items.length ? `<div class="alert-list">${items.map((alert) => `
        <div class="alert-item ${alert.level || "info"}">
          <strong>${alert.title || "SRM parking update"}</strong>
          <span>${alert.message || ""}</span>
        </div>
      `).join("")}</div>` : `<div class="fallback-panel compact"><span>No active notifications.</span></div>`}
    </section>
  `;
}

function buildLearningPanel(state) {
  const learning = state.learning || {};
  const rules = learning.llm_memory_rules || [];
  const blocked = learning.blocked_routes || [];
  const recentRouteCounts = learning.recent_route_counts || {};
  const repeatedRoutes = Object.entries(recentRouteCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);
  const current = state.latest_decision || {};
  const currentRoute = current.action === "redirect" ? `${current.from}->${current.to}` : "";
  const blockedConflict = currentRoute && blocked.includes(currentRoute);
  const learningMessage = blocked.length || repeatedRoutes.length
    ? "Repeated route penalized. The agent will avoid weak routes and prefer stronger alternatives."
    : (learning.latest_learning_insight || "Learning state will update as rewards and route outcomes are recorded.");
  const conflictNote = blockedConflict
    ? `Emergency override allowed ${currentRoute} for this live step; learning will still avoid it in future normal steps.`
    : "";
  return `
    <section class="learning-panel">
      <div class="board-header">
        <div>
          <h2>Learning Overlay</h2>
          <span>${learningMessage}</span>
        </div>
      </div>
      <div class="learning-grid">
        <div><span>Reward Trend</span><strong>${Number(learning.recent_reward_avg || 0).toFixed(2)}</strong></div>
        <div><span>Blocked Routes</span><strong>${blocked.length}</strong></div>
        <div><span>LLM Memory Rules</span><strong>${rules.length}</strong></div>
        <div><span>Route Diversity</span><strong>${repeatedRoutes.length ? "Active" : "Open"}</strong></div>
      </div>
      ${blockedConflict ? `<div class="learning-warning">${escapeHtml(conflictNote)}</div>` : ""}
      ${(blocked.length || rules.length || repeatedRoutes.length) ? `<div class="route-tags">
        ${blocked.slice(0, 4).map((route) => `<span class="route-tag ${route === currentRoute ? "caution" : "bad"}">${route === currentRoute ? "Emergency override" : "Inefficient"} ${route}</span>`).join("")}
        ${rules.slice(0, 4).map((rule) => {
          const route = rule.route_key || `${rule.from || "-"}->${rule.to || "-"}`;
          const avoid = Number(rule.strength || 0) < 0 || Number(rule.avoid_count || 0) > Number(rule.prefer_count || 0);
          return `<span class="route-tag ${avoid ? "bad" : "good"}">${avoid ? "Avoid" : "Prefer"} ${route}</span>`;
        }).join("")}
        ${repeatedRoutes.map(([route, count]) => `<span class="route-tag caution">Recently used ${route} x${count}</span>`).join("")}
      </div>` : ""}
    </section>
  `;
}

function buildBlockLayout(blockEntries, containerWidth) {
  const gap = 24;
  const blockWidth = 328;
  const blockHeight = 286;
  const columns = Math.max(1, Math.floor((containerWidth + gap) / (blockWidth + gap)));
  return blockEntries.reduce((acc, [name], index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    acc[name] = {
      x: column * (blockWidth + gap),
      y: row * (blockHeight + gap),
      width: blockWidth,
      height: blockHeight,
    };
    return acc;
  }, {});
}

function vehicleScreenPosition(vehicle, layout) {
  const block = layout[vehicle.block];
  if (!block) return { left: 0, top: 0 };
  const slot = Math.max(1, Number(vehicle.slot || vehicle.id || 1));
  const fallbackPosition = {
    x: 0.08 + ((slot - 1) % 8) * 0.105,
    y: 0.16 + (Math.floor((slot - 1) / 8) % 4) * 0.18,
  };
  const position = vehicle.position && Number.isFinite(Number(vehicle.position.x)) && Number.isFinite(Number(vehicle.position.y))
    ? vehicle.position
    : fallbackPosition;
  return {
    left: block.x + 28 + Number(position.x || 0) * (block.width - 56),
    top: block.y + 124 + Number(position.y || 0) * (block.height - 166),
  };
}

function pickVisibleVehicles(vehicles) {
  const byBlock = new Map();
  vehicles.filter((vehicle) => vehicle.status !== "exited").forEach((vehicle) => {
    const bucket = byBlock.get(vehicle.block) || [];
    bucket.push(vehicle);
    byBlock.set(vehicle.block, bucket);
  });
  const visible = [];
  byBlock.forEach((items) => {
    const userVehicles = items
      .filter((vehicle) => vehicle.user_type && vehicle.user_type !== "simulated")
      .sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const simulatedVehicles = items
      .filter((vehicle) => !vehicle.user_type || vehicle.user_type === "simulated")
      .sort((a, b) => (a.slot || 0) - (b.slot || 0));
    const limit = 14;
    if (items.length <= limit) {
      visible.push(...userVehicles, ...simulatedVehicles);
      return;
    }
    visible.push(...userVehicles.slice(0, limit));
    const remaining = Math.max(0, limit - Math.min(limit, userVehicles.length));
    const stride = simulatedVehicles.length / Math.max(1, remaining);
    for (let index = 0; index < remaining; index += 1) {
      const candidate = simulatedVehicles[Math.floor(index * stride)];
      if (candidate) visible.push(candidate);
    }
  });
  return visible;
}

function buildVehicleOverlay(vehicles, layout, state = latestState) {
  const affected = new Set(affectedVehicleIds(state));
  const important = vehicles
    .filter((vehicle) =>
      vehicle.user_type && vehicle.user_type !== "simulated" ||
      vehicle.status === "redirecting" ||
      affected.has(String(vehicle.id)) ||
      affected.has(String(vehicle.number))
    )
    .slice(0, 32);
  return important.map((vehicle) => {
    const pos = vehicleScreenPosition(vehicle, layout);
    const isUser = vehicle.user_type && vehicle.user_type !== "simulated";
    const label = vehicle.number || vehicle.id || "Vehicle";
    const name = vehicle.name ? ` · ${vehicle.name}` : "";
    return `
      <button class="vehicle-ghost ${isUser ? "user" : "simulated"} status-${vehicle.status || "parked"}"
        style="left:${pos.left}px; top:${pos.top}px"
        data-vehicle-id="${escapeHtml(vehicle.id)}"
        title="${escapeHtml(`${label}${name} · ${vehicle.user_type || "simulated"} · ${vehicle.block || "-"}`)}">
        ${realisticVehicleMarkup(vehicle.type, label, "mini")}
        <span class="vehicle-overlay-label">${escapeHtml(label)}</span>
      </button>
    `;
  }).join("");
}

function buildUserActivityStrip(state) {
  const users = (state.user_vehicles || [])
    .slice()
    .sort((a, b) => {
      const aActive = a.status !== "exited" ? 0 : 1;
      const bActive = b.status !== "exited" ? 0 : 1;
      if (aActive !== bActive) return aActive - bActive;
      return String(b.updated_at || b.timestamp || b.id || "").localeCompare(String(a.updated_at || a.timestamp || a.id || ""));
    })
    .slice(0, 8);
  const events = (state.events || []).filter((event) => event.user_type !== "simulated").slice(-6).reverse();
  return `
    <section class="user-sync-panel">
      <div>
        <span>User App Sync</span>
        <strong>${formatNumber(users.length)} recent user vehicle${users.length === 1 ? "" : "s"}</strong>
        <small>New web-app entries are highlighted on the map and mirrored in the dashboard.</small>
      </div>
      <div class="user-sync-list">
        ${users.length ? users.map((vehicle, index) => `
          <button class="user-sync-item ${index === 0 ? "fresh" : ""}" data-vehicle-id="${escapeHtml(vehicle.id)}">
            <b>${escapeHtml(vehicle.number || vehicle.id)}</b>
            <span>${escapeHtml(vehicle.user_type || "user")} · ${escapeHtml(vehicle.status || "parked")} · ${escapeHtml(vehicle.block || "-")}</span>
          </button>
        `).join("") : `<div class="user-sync-empty">No user entries yet. Submit from the user app to see it appear here instantly.</div>`}
      </div>
      <div class="user-event-mini">
        ${events.length ? events.map((event) => `<span>${escapeHtml(event.vehicle_number || "Vehicle")} ${escapeHtml(event.event || "")} → ${escapeHtml(event.to_block || event.block || event.to_gate || "-")}</span>`).join("") : "<span>Waiting for user entry / exit events</span>"}
      </div>
    </section>
  `;
}

function buildVehicleListPanel(state, activeVehicles) {
  const affected = new Set(affectedVehicleIds(state));
  const prioritized = activeVehicles.filter((vehicle) =>
    affected.has(String(vehicle.id)) ||
    affected.has(String(vehicle.number)) ||
    vehicle.status === "redirecting" ||
    (vehicle.user_type && vehicle.user_type !== "simulated")
  );
  const rows = showAllVehicles ? activeVehicles.slice(0, 140) : prioritized.slice(0, 24);
  return `
    <section class="vehicle-list-panel">
      <div class="board-header">
        <div>
          <h2>${showAllVehicles ? "All Visible Vehicles" : "Moving / Affected Vehicles"}</h2>
          <span>${showAllVehicles ? "Showing a capped backend list for performance." : "Default view stays focused on active movement and user vehicles."}</span>
        </div>
        <button class="secondary-button" data-toggle-vehicles="true">${showAllVehicles ? "Show moving only" : "Show all vehicles"}</button>
      </div>
      ${rows.length ? `<div class="vehicle-list-grid">${rows.map((vehicle) => `
        <button class="vehicle-list-item ${vehicle.user_type && vehicle.user_type !== "simulated" ? "user" : "simulated"}" data-vehicle-id="${escapeHtml(vehicle.id)}">
          ${realisticVehicleMarkup(vehicle.type, vehicle.number || vehicle.id, "mini")}
          <span>${escapeHtml(vehicle.number || vehicle.id)}</span>
          <small>${escapeHtml(vehicle.user_type || "simulated")} · ${escapeHtml(vehicle.status || "parked")} · ${escapeHtml(vehicle.block || "-")}</small>
        </button>
      `).join("")}</div>` : `<div class="fallback-panel compact"><span>No moving vehicles in the current backend step.</span></div>`}
    </section>
  `;
}

function buildArrowMarkup(action, layout, boardWidth, boardHeight) {
  if (!action || action.type !== "redirect" || !layout[action.from] || !layout[action.to]) return "";
  const source = layout[action.from];
  const destination = layout[action.to];
  const x1 = source.x + source.width / 2;
  const y1 = source.y + source.height / 2;
  const x2 = destination.x + destination.width / 2;
  const y2 = destination.y + destination.height / 2;
  const vehicleCount = Math.max(0, Number(action.vehicles || 0));
  return `
    <svg class="flow-overlay" viewBox="0 0 ${boardWidth} ${boardHeight}" preserveAspectRatio="none">
      <defs>
        <marker id="flow-arrow" markerWidth="14" markerHeight="14" refX="8" refY="4" orient="auto">
          <path d="M0,0 L0,8 L8,4 z" fill="#8ad8ff"></path>
        </marker>
      </defs>
      <line
        x1="${x1}"
        y1="${y1}"
        x2="${x2}"
        y2="${y2}"
        stroke="#8ad8ff"
        stroke-width="4"
        stroke-linecap="round"
        marker-end="url(#flow-arrow)"
      ></line>
    </svg>
    ${Array.from({ length: vehicleCount }, (_, index) => `
      <div
        class="route-particle ${index < Number(action.car_vehicles || 0) ? "car" : "bike"}"
        style="--x1:${x1}px; --y1:${y1}px; --x2:${x2}px; --y2:${y2}px; --delay:${index * 0.16}s;"
        title="Moving vehicle ${index + 1} of ${vehicleCount}"
      >${realisticVehicleMarkup(index < Number(action.car_vehicles || 0) ? "car" : "bike", "", "moving")}</div>
    `).join("")}
  `;
}

function buildGateMarkup(gates, boardWidth, boardHeight) {
  return Object.entries(gates || {}).map(([name, gate]) => {
    const x = Number(gate.x || 0.5) * boardWidth;
    const y = Number(gate.y || 0.5) * boardHeight;
    return `<div class="gate-marker" data-gate="${name}" style="left:${x}px; top:${y}px;"><span>${name}</span><small>ENTRY</small></div>`;
  }).join("");
}

function buildEntryFlow(block) {
  const entry = Number(block.entry || 0);
  const exit = Number(block.exit || 0);
  return `
    <div class="flow-counts">
      <span>In ${formatNumber(entry)}</span>
      <span>Out ${formatNumber(exit)}</span>
    </div>
  `;
}

function buildLiveMovementPanel(state) {
  const stats = state.vehicle_stats || {};
  const lastEvent = (state.events || []).slice(-1)[0];
  const action = activeRouteAction(state);
  const visibleMoving = action?.type === "redirect" ? Number(action.vehicles || 0) : Number(stats.redirecting || 0);
  const blockTotals = Object.values(state.blocks || {}).reduce((acc, block) => {
    acc.entry += Number(block.entry || 0);
    acc.exit += Number(block.exit || 0);
    return acc;
  }, { entry: 0, exit: 0 });
  const entering = Number(stats.entering || 0) || blockTotals.entry;
  const exiting = Number(stats.exiting || 0) || blockTotals.exit;
  return `
    <section class="movement-panel">
      <div>
        <span>Moving Now</span>
        <strong>${formatNumber(visibleMoving)}</strong>
      </div>
      <div>
        <span>Entering</span>
        <strong>${formatNumber(entering)}</strong>
      </div>
      <div>
        <span>Exiting</span>
        <strong>${formatNumber(exiting)}</strong>
      </div>
      <div>
        <span>Last Event</span>
        <strong>${lastEvent ? `${lastEvent.vehicle_number || "Vehicle"} ${lastEvent.event}` : "Waiting"}</strong>
      </div>
    </section>
  `;
}

function buildPreviewSlots(blockName, block, vehicles) {
  const previewSlots = 36;
  const capacity = Math.max(0, Number(block.capacity || 0));
  const occupied = Math.min(capacity, Math.max(0, Number(block.occupied || 0)));
  const occupiedPreview = capacity ? Math.min(previewSlots, Math.round((occupied / capacity) * previewSlots)) : 0;
  const vehiclesHere = vehicles.filter((vehicle) => vehicle.block === blockName).slice(0, occupiedPreview);
  return Array.from({ length: previewSlots }, (_, index) => {
    const filled = index < occupiedPreview;
    const vehicle = vehiclesHere[index];
    const icon = vehicle ? realisticVehicleMarkup(vehicle.type, vehicle.number || vehicle.id, "slot-vehicle") : "";
    const userClass = vehicle?.user_type && vehicle.user_type !== "simulated" ? " user-vehicle" : " simulated-vehicle";
    const statusClass = vehicle?.status ? ` status-${vehicle.status}` : "";
    const vehicleAttr = vehicle ? ` data-vehicle-id="${vehicle.id}" title="${vehicle.number || vehicle.id} · ${vehicle.user_type} · ${vehicle.status}"` : "";
    return `<div class="slot ${filled ? "occupied" : "free"}${filled ? userClass : ""}${statusClass}"${vehicleAttr}>${icon || (filled ? "<span class=\"slot-fill\"></span>" : "")}</div>`;
  }).join("");
}

function buildBlockRouteBadge(name, latestAction) {
  if (!latestAction || latestAction.type !== "redirect") return "";
  if (latestAction.from === name) return `<div class="route-role source">Source</div>`;
  if (latestAction.to === name) return `<div class="route-role destination">Destination</div>`;
  return "";
}

function buildDetailSlots(blockName, block, vehicles) {
  const capacity = Math.max(0, Number(block.capacity || 0));
  const occupiedSet = new Map(vehicles.filter((vehicle) => vehicle.block === blockName).map((vehicle) => [vehicle.slot, vehicle]));
  const carSlots = Number(block.car_slots || capacity);
  return Array.from({ length: capacity }, (_, index) => {
    const slotNumber = index + 1;
    const vehicle = occupiedSet.get(slotNumber);
    const slotType = slotNumber <= carSlots ? "car-slot" : "bike-slot";
    const icon = vehicle ? realisticVehicleMarkup(vehicle.type, vehicle.number || vehicle.id, "detail-vehicle") : "";
    const label = vehicle ? `${vehicle.number || vehicle.id}: ${vehicle.type} ${vehicle.status || "parked"}` : "Free";
    const userClass = vehicle?.user_type && vehicle.user_type !== "simulated" ? " user-vehicle" : " simulated-vehicle";
    const statusClass = vehicle?.status ? ` status-${vehicle.status}` : "";
    const vehicleAttr = vehicle ? ` data-vehicle-id="${vehicle.id}"` : "";
    return `
      <div class="detail-slot ${slotType} ${vehicle ? `filled${userClass}${statusClass}` : "empty"}" title="Slot ${slotNumber}: ${label}"${vehicleAttr}>
        <span class="slot-id">${slotNumber}</span>
        <span class="slot-icon">${icon}</span>
      </div>
    `;
  }).join("");
}

function buildDetailPanel(state) {
  if (!selectedBlock || !state.blocks[selectedBlock]) return "";
  const block = state.blocks[selectedBlock];
  const vehicles = state.vehicles || [];
  const occupied = Number(block.occupied || 0);
  const capacity = Number(block.capacity || 0);
  const freeSlots = Math.max(0, capacity - occupied);
  const level = occupancyLevel(occupied, capacity);
  const blockVehicles = vehicles.filter((vehicle) => vehicle.block === selectedBlock);
  const selectedVehicle = vehicles.find((vehicle) => String(vehicle.id) === String(selectedVehicleId));
  const cars = blockVehicles.filter((vehicle) => vehicle.type === "car").length;
  const bikes = blockVehicles.filter((vehicle) => vehicle.type === "bike").length;

  return `
    <section class="detail-panel ${level}">
      <div class="detail-header">
        <div>
          <div class="detail-kicker">Interactive Block View</div>
          <h3>${selectedBlock}</h3>
          <p>Every slot below is state-driven from the backend. Filled slots contain real rendered cars or bikes.</p>
        </div>
        <button class="close-detail" data-close-detail="true" aria-label="Close block detail">×</button>
      </div>
      <div class="detail-metrics">
        <div><span>Occupied</span><strong>${formatNumber(occupied)}</strong></div>
        <div><span>Free</span><strong>${formatNumber(freeSlots)}</strong></div>
        <div><span>Car Slots</span><strong>${formatNumber(block.car_slots || cars)}</strong></div>
        <div><span>Bike Slots</span><strong>${formatNumber(block.bike_slots || bikes)}</strong></div>
      </div>
      <div class="detail-legend">
        <div><span class="legend-dot car"></span>Car slot</div>
        <div><span class="legend-dot bike"></span>Bike slot</div>
        <div><span class="legend-dot free"></span>Free slot</div>
      </div>
      ${selectedVehicle ? `
        <div class="vehicle-inspector">
          <span>Selected Vehicle</span>
          <strong>${selectedVehicle.number || selectedVehicle.id}</strong>
          <small>${selectedVehicle.user_type || "simulated"} · ${selectedVehicle.type || "-"} · ${selectedVehicle.status || "parked"} · ${selectedVehicle.block || "-"}</small>
        </div>
      ` : ""}
      <div class="detail-grid">${buildDetailSlots(selectedBlock, block, vehicles)}</div>
    </section>
  `;
}

function render(state, error = "") {
  const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
  const blockEntries = Object.entries(state.blocks || {});
  if (selectedBlock && !state.blocks[selectedBlock]) selectedBlock = null;

  const containerWidth = Math.max(1160, window.innerWidth - 88);
  const layout = buildBlockLayout(blockEntries, containerWidth);
  const columns = Math.max(1, Math.floor((containerWidth + 24) / (328 + 24)));
  const rows = Math.max(1, Math.ceil(blockEntries.length / columns));
  const boardHeight = rows * 286 + Math.max(0, rows - 1) * 24;
  const latestAction = activeRouteAction(state);
  const activeVehicles = (state.vehicles || []).filter((vehicle) => vehicle.status !== "exited");
  const visibleVehicles = pickVisibleVehicles(activeVehicles);
  const totalVehicles = activeVehicles.length;
  const totalCapacity = blockEntries.reduce((sum, [, block]) => sum + Math.max(0, Number(block.capacity || 0)), 0);
  const totalOccupied = blockEntries.reduce((sum, [, block]) => {
    const capacity = Math.max(0, Number(block.capacity || 0));
    return sum + Math.min(capacity, Math.max(0, Number(block.occupied || 0)));
  }, 0);
  const totalFree = Math.max(0, totalCapacity - totalOccupied);

  app.innerHTML = `
    <div class="app-shell">
      <header class="hero-panel">
        <div class="hero-copy">
          <div class="eyebrow">SRM Smart Parking</div>
          <h1>Interactive Parking Simulation</h1>
          <p>
            Click any block to open its real slot layout. Cars and bikes are mapped from backend vehicle state,
            while redirects highlight and animate across the campus view.
          </p>
        </div>
        <div class="hero-meta">
          <div class="status-pill live"><span class="live-dot"></span>Live now</div>
          <div class="status-pill">Step ${state.step || 0}</div>
          <div class="status-pill muted">${formatTimestamp(state.updated_at)}</div>
          <div class="status-pill accent">${latestAction ? `${String(latestAction.type).toUpperCase()} ${latestAction.vehicles}` : "Monitoring"}</div>
        </div>
      </header>

      ${error ? `<div class="fallback-panel"><strong>API unavailable.</strong><span>${error}</span></div>` : ""}
      <div class="view-toggle">
        <button class="${viewMode === "basic" ? "active" : ""}" data-view-mode="basic">Basic View</button>
        <button class="${viewMode === "advanced" ? "active" : ""}" data-view-mode="advanced">Advanced View</button>
      </div>
      ${buildDecisionPanel(state)}
      ${buildLiveMovementPanel(state)}
      ${buildMiniMap(state)}
      ${buildLiveRouteStage(state)}
      ${buildNotificationStrip(state)}
      ${viewMode === "advanced" ? buildUserActivityStrip(state) : ""}
      ${viewMode === "advanced" ? buildLLMRouterPanel(state) : ""}
      ${viewMode === "advanced" ? buildAlertsPanel(state) : ""}

      <section class="summary-strip">
        <div class="summary-card pressure-${occupancyLevel(totalOccupied, totalCapacity)}">
          <span>Campus Fill Level</span>
          <strong>${totalCapacity ? Math.round((totalOccupied / totalCapacity) * 100) : 0}%</strong>
          <small>${totalFree > 0 ? `${formatNumber(totalFree)} slots still available` : "Campus is full"}</small>
        </div>
        <div class="summary-card">
          <span>Parking Blocks</span>
          <strong>${blockEntries.length}</strong>
          <small>Dynamic block rendering</small>
        </div>
        <div class="summary-card">
          <span>Total Occupied</span>
          <strong>${formatNumber(totalOccupied)}</strong>
          <small>Live occupied slots</small>
        </div>
        <div class="summary-card">
          <span>Total Free</span>
          <strong>${formatNumber(totalFree)}</strong>
          <small>Campus-wide free capacity</small>
        </div>
      </section>

      ${buildZoneSlotOverview(state)}

      ${viewMode === "advanced" ? `<section class="board-card">
        <div class="board-header">
          <div>
          <h2>SRM Parking Blocks</h2>
            <span>Road lanes, gates, and real vehicle sprites show live parking movement. Click to inspect real slots.</span>
          </div>
          <div class="legend">
            <div class="legend-item"><span class="legend-dot high"></span>High occupancy</div>
            <div class="legend-item"><span class="legend-dot medium"></span>Medium occupancy</div>
            <div class="legend-item"><span class="legend-dot low"></span>Low occupancy</div>
          </div>
        </div>

        <div class="board realistic-board" style="height:${boardHeight}px">
          <div class="board-road-layer" aria-hidden="true">
            <span class="board-road horizontal primary"></span>
            <span class="board-road horizontal secondary"></span>
            <span class="board-road vertical primary"></span>
            <span class="board-road vertical secondary"></span>
          </div>
          ${buildArrowMarkup(latestAction, layout, containerWidth, boardHeight)}
          ${buildGateMarkup(state.gates || {}, containerWidth, boardHeight)}
          ${buildVehicleOverlay(activeVehicles, layout, state)}
          ${blockEntries.map(([name, block]) => {
            const blockLayout = layout[name];
            const capacity = Math.max(0, Number(block.capacity || 0));
            const occupied = Math.min(capacity, Math.max(0, Number(block.occupied || 0)));
            const freeSlots = Math.max(0, capacity - occupied);
            const level = occupancyLevel(occupied, capacity);
            const activeRedirect = latestAction && (latestAction.from === name || latestAction.to === name) ? " active-block" : "";
            const sourceClass = latestAction?.from === name ? " source-block" : "";
            const destinationClass = latestAction?.to === name ? " destination-block" : "";
            const inactiveClass = latestAction?.type === "redirect" && latestAction.from !== name && latestAction.to !== name ? " inactive-route" : "";
            const selected = selectedBlock === name ? " selected-block" : "";
            const progress = capacity ? Math.round((occupied / capacity) * 100) : 0;
            const carSlots = Number(block.car_slots || capacity);
            const bikeSlots = Number(block.bike_slots || Math.max(0, capacity - carSlots));
            return `
              <button class="block-card ${level}${activeRedirect}${sourceClass}${destinationClass}${inactiveClass}${selected}" data-block="${name}" style="transform:translate(${blockLayout?.x || 0}px, ${blockLayout?.y || 0}px); width:${blockLayout?.width || 328}px; height:${blockLayout?.height || 286}px;">
                ${buildBlockRouteBadge(name, latestAction)}
                <div class="block-top">
                  <div>
                    <div class="block-kicker">Parking Block</div>
                    <div class="block-name">${name}</div>
                  </div>
                  <div class="occupancy-chip">${progress}%</div>
                </div>
                <div class="gate-row">
                  <div class="gate-label">Live Flow</div>
                  ${buildEntryFlow(block)}
                </div>
                <div class="slot-grid preview">${buildPreviewSlots(name, block, visibleVehicles)}</div>
                <div class="slot-breakdown">
                  <span>Car Slots ${formatNumber(carSlots)}</span>
                  <span>Bike Slots ${formatNumber(bikeSlots)}</span>
                </div>
                <div class="block-footer">
                  <div><span>Occupied</span><strong>${formatNumber(occupied)}</strong></div>
                  <div><span>Free</span><strong>${formatNumber(freeSlots)}</strong></div>
                  <div><span>Capacity</span><strong>${formatNumber(capacity)}</strong></div>
                </div>
                <div class="block-progress"><div style="width:${progress}%"></div></div>
              </button>
            `;
          }).join("")}
        </div>
      </section>` : ""}

      ${viewMode === "advanced" ? buildVehicleListPanel(state, activeVehicles) : ""}
      ${buildDetailPanel({ ...state, vehicles: activeVehicles })}
      ${viewMode === "advanced" ? buildLearningPanel(state) : ""}

      ${viewMode === "advanced" ? `<details class="action-card collapsible" open>
        <summary>
          <div>
            <h2>Redirect Timeline</h2>
            <span>Last 5 entries are shown by default. Expand when judges ask for deeper history.</span>
          </div>
        </summary>
        ${(state.actions || []).length ? `
          <ul class="action-list">
            ${state.actions.slice().reverse().slice(0, timelineExpanded ? 20 : 5).map((action) => `
              <li class="${latestAction && latestAction.step === action.step ? "current" : ""}">
                <div class="action-step">${formatTimestamp(action.timestamp)} · Step ${action.step}</div>
                <div class="action-route">${String(action.type).toUpperCase()} ${action.vehicles}: ${action.from} → ${action.to}</div>
                <div class="action-meta">IDs ${(action.vehicle_ids || []).slice(0, 4).join(", ") || "-"} · ${shortText(action.reason || "Agent route execution", 110)}</div>
              </li>
            `).join("")}
          </ul>
          ${(state.actions || []).length > 5 ? `<button class="secondary-button timeline-more" data-toggle-timeline="true">${timelineExpanded ? "Show last 5" : "View more"}</button>` : ""}
        ` : `
          <div class="fallback-panel compact">
            <span>No redirect actions yet. The next backend redirect will animate across the source and destination blocks.</span>
          </div>
        `}
      </details>` : ""}
    </div>
  `;

  // Motion is intentionally contained inside the map and route lane for a cleaner demo.
  window.requestAnimationFrame(() => window.scrollTo({ top: scrollTop, behavior: "auto" }));
}

function safeRender(state, error = "") {
  try {
    render(state, error);
  } catch (renderError) {
    console.error("Frontend render failed", renderError);
    app.innerHTML = `
      <div class="app-shell">
        <section class="fallback-panel">
          <strong>Frontend render recovered.</strong>
          <span>${escapeHtml(renderError?.message || "A vehicle or block payload was missing optional visual data.")}</span>
          <small>The backend is still running. Refresh once or switch back to Basic View.</small>
        </section>
      </div>
    `;
  }
}

function animateRedirect(action, oldState, currentState, layout) {
  if (!action || action.type !== "redirect" || action.step === lastAnimatedStep) return;
  lastAnimatedStep = action.step;
  const board = document.querySelector(".board");
  if (!board || !layout[action.from] || !layout[action.to]) return;

  const count = Math.max(0, Number(action.vehicles || 0));
  if (!count) return;
  const fromEl = document.querySelector(`[data-block="${cssAttr(action.from)}"]`);
  const toEl = document.querySelector(`[data-block="${cssAttr(action.to)}"]`);
  if (fromEl && toEl) animateVehicles(count, fromEl, toEl);
}

function animateVehicleEvents(state, layout) {
  const recentEvents = (state.events || []).slice(-12);
  recentEvents.forEach((event) => {
    if (!event.id || animatedEventIds.has(event.id)) return;
    if (!["entry", "redirect", "exit"].includes(event.event)) return;
    animatedEventIds.add(event.id);
    const label = `${event.vehicle_number || "Vehicle"} · ${event.user_type || ""}`;
    if (event.event === "entry") {
      const fromEl = document.querySelector(`.board [data-gate="${cssAttr(event.gate || event.from_gate)}"]`);
      const toEl = document.querySelector(`[data-block="${cssAttr(event.to_block || event.block)}"]`);
      if (fromEl && toEl) animateLabeledVehicle(label, fromEl, toEl);
    } else if (event.event === "exit") {
      const fromEl = document.querySelector(`[data-block="${cssAttr(event.from_block || event.block)}"]`);
      const toEl = document.querySelector(`.board [data-gate="${cssAttr(event.to_gate || event.gate)}"]`);
      if (fromEl && toEl) animateLabeledVehicle(label, fromEl, toEl, true);
    } else if (event.event === "redirect") {
      const fromEl = document.querySelector(`[data-block="${cssAttr(event.from_block || event.from)}"]`);
      const toEl = document.querySelector(`[data-block="${cssAttr(event.to_block || event.to)}"]`);
      if (fromEl && toEl) animateLabeledVehicle(label, fromEl, toEl);
    }
  });
  if (animatedEventIds.size > 80) {
    animatedEventIds = new Set(Array.from(animatedEventIds).slice(-40));
  }
}

function animateLabeledVehicle(label, fromEl, toEl, removeAtEnd = false) {
  const vehicle = document.createElement("div");
  vehicle.className = "vehicle-flow-label";
  vehicle.innerHTML = `${realisticVehicleMarkup("car", label, "moving")}<span>${label}</span>`;
  const from = fromEl.getBoundingClientRect();
  const to = toEl.getBoundingClientRect();
  const angle = Math.atan2((to.top + to.height / 2) - (from.top + from.height / 2), (to.left + to.width / 2) - (from.left + from.width / 2)) * 180 / Math.PI;
  vehicle.style.left = `${from.left + from.width / 2}px`;
  vehicle.style.top = `${from.top + from.height / 2}px`;
  vehicle.style.setProperty("--angle", `${angle}deg`);
  document.body.appendChild(vehicle);
  setTimeout(() => {
    vehicle.style.left = `${to.left + to.width / 2}px`;
    vehicle.style.top = `${to.top + to.height / 2}px`;
    if (removeAtEnd) vehicle.style.opacity = "0.25";
  }, 50);
  setTimeout(() => vehicle.remove(), 2200);
}

function cssAttr(value) {
  return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function animateVehicles(count, fromEl, toEl) {
  for (let i = 0; i < count; i++) {
    const vehicle = document.createElement("div");
    vehicle.className = "vehicle-dot";
    vehicle.innerHTML = realisticVehicleMarkup("car", `Moving vehicle ${i + 1}`, "moving");

    const from = fromEl.getBoundingClientRect();
    const to = toEl.getBoundingClientRect();
    const offset = (i - (count - 1) / 2) * 9;
    const angle = Math.atan2((to.top + to.height / 2) - (from.top + from.height / 2), (to.left + to.width / 2) - (from.left + from.width / 2)) * 180 / Math.PI;

    vehicle.style.position = "fixed";
    vehicle.style.left = `${from.left + from.width / 2 + offset}px`;
    vehicle.style.top = `${from.top + from.height / 2 + offset}px`;
    vehicle.style.setProperty("--angle", `${angle}deg`);
    vehicle.style.transition = "left 1.8s cubic-bezier(0.22, 0.75, 0.28, 1), top 1.8s cubic-bezier(0.22, 0.75, 0.28, 1), opacity 1.8s ease";

    document.body.appendChild(vehicle);

    setTimeout(() => {
      vehicle.style.left = `${to.left + to.width / 2 + offset}px`;
      vehicle.style.top = `${to.top + to.height / 2 + offset}px`;
    }, 50 + i * 35);

    setTimeout(() => {
      vehicle.remove();
    }, 2200 + i * 35);
  }
}

async function fetchState() {
  try {
    const response = await fetch(`${API_BASE}/client-state?_ts=${Date.now()}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const payload = await response.json();
    const shared = { ...(payload || {}), ...(payload.current_state || {}) };
    previousState = latestState;
    const nextState = {
      step: shared.step || 0,
      updated_at: shared.updated_at || "",
      blocks: shared.blocks || {},
      vehicles: shared.vehicles || [],
      simulated_vehicles: shared.simulated_vehicles || payload.simulated_vehicles || [],
      user_vehicles: shared.user_vehicles || payload.user_vehicles || [],
      users: shared.users || payload.users || [],
      events: shared.events || payload.events || [],
      gates: shared.gates || payload.gates || {},
      vehicle_stats: shared.vehicle_stats || payload.vehicle_stats || {},
      actions: shared.actions || [],
      movement_log: shared.movement_log || payload.movement_log || [],
      alerts: shared.alerts || payload.alerts || [],
      latest_decision: shared.latest_decision || payload.latest_decision || {},
      latest_result: shared.latest_result || payload.latest_result || {},
      event_context: shared.event_context || payload.event_context || {},
      reasoning_summary: shared.reasoning_summary || payload.reasoning_summary || {},
      llm_usage_summary: shared.llm_usage_summary || payload.llm_usage_summary || {},
      decision_reason: shared.decision_reason || payload.decision_reason || "",
      agent_thought: shared.agent_thought || payload.agent_thought || "",
      learning: shared.learning || payload.learning || {},
      llm: shared.llm || payload.llm || {},
    };
    const nextKey = JSON.stringify({
      step: nextState.step,
      updated_at: nextState.updated_at,
      vehicleCount: nextState.vehicles.length,
      userVehicleCount: nextState.user_vehicles.length,
      userVehicles: nextState.user_vehicles.slice(-8).map((vehicle) => `${vehicle.id}:${vehicle.number}:${vehicle.status}:${vehicle.block}:${vehicle.name}`).join("|"),
      eventCount: nextState.events.length,
      actionCount: nextState.actions.length,
      alertCount: nextState.alerts.length,
      decision: nextState.latest_decision,
      latestAction: nextState.actions?.[nextState.actions.length - 1],
      selectedBlock,
      viewMode,
      showAllVehicles,
      timelineExpanded,
    });
    latestState = nextState;
    if (nextKey !== lastRenderKey) {
      lastRenderKey = nextKey;
      safeRender(latestState, "");
    }
  } catch (error) {
    safeRender(latestState, error?.message || "Unable to fetch parking state.");
  }
}

app.addEventListener("click", (event) => {
  const modeButton = event.target.closest("[data-view-mode]");
  if (modeButton) {
    viewMode = modeButton.getAttribute("data-view-mode") || "basic";
    try {
      localStorage.setItem("parkingViewMode", viewMode);
    } catch (error) {
      // Storage can be blocked in embedded browsers; the UI should still work.
    }
    safeRender(latestState, "");
    return;
  }
  const close = event.target.closest("[data-close-detail='true']");
  if (close) {
    selectedBlock = null;
    selectedVehicleId = null;
    safeRender(latestState, "");
    return;
  }
  const vehicleToggle = event.target.closest("[data-toggle-vehicles='true']");
  if (vehicleToggle) {
    showAllVehicles = !showAllVehicles;
    safeRender(latestState, "");
    return;
  }
  const timelineToggle = event.target.closest("[data-toggle-timeline='true']");
  if (timelineToggle) {
    timelineExpanded = !timelineExpanded;
    safeRender(latestState, "");
    return;
  }
  const vehicleEl = event.target.closest("[data-vehicle-id]");
  if (vehicleEl) {
    selectedVehicleId = vehicleEl.getAttribute("data-vehicle-id");
    const vehicle = (latestState.vehicles || []).find((item) => String(item.id) === String(selectedVehicleId));
    if (vehicle?.block) selectedBlock = vehicle.block;
    safeRender(latestState, "");
    return;
  }
  const blockButton = event.target.closest("[data-block]");
  if (blockButton) {
    selectedBlock = blockButton.getAttribute("data-block");
    safeRender(latestState, "");
  }
});

window.addEventListener("resize", () => safeRender(latestState, ""));
safeRender(latestState, "");
fetchState();
window.setInterval(fetchState, 1000);
