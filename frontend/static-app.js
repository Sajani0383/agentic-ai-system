const API_BASE = "http://127.0.0.1:8000";
const EMPTY_STATE = {
  step: 0,
  updated_at: "",
  blocks: {},
  vehicles: [],
  actions: [],
  movement_log: [],
  alerts: [],
  latest_decision: {},
  decision_reason: "",
  agent_thought: "",
  learning: {},
  llm: {},
};

const app = document.getElementById("app");
let latestState = { ...EMPTY_STATE };
let previousState = { ...EMPTY_STATE };
let selectedBlock = null;
let lastAnimatedStep = -1;
let lastRenderKey = "";

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

function buildMiniMap(state) {
  const entries = Object.entries(state.blocks || {});
  const latestAction = state.actions?.length ? state.actions[state.actions.length - 1] : null;
  const nodes = entries.map(([name, block], index) => {
    const occupied = Number(block.occupied || 0);
    const capacity = Number(block.capacity || 0);
    const pos = campusPosition(name, index, entries.length);
    const level = occupancyLevel(occupied, capacity);
    const source = latestAction?.from === name ? " source-node" : "";
    const destination = latestAction?.to === name ? " destination-node" : "";
    return `<button class="map-node ${level}${source}${destination}" data-block="${name}" style="left:${pos.left}%; top:${pos.top}%;" title="${name}: ${occupied}/${capacity} occupied">${name}</button>`;
  }).join("");
  let arrow = "";
  if (latestAction?.type === "redirect") {
    const sourceIndex = entries.findIndex(([name]) => name === latestAction.from);
    const destIndex = entries.findIndex(([name]) => name === latestAction.to);
    if (sourceIndex >= 0 && destIndex >= 0) {
      const source = campusPosition(latestAction.from, sourceIndex, entries.length);
      const dest = campusPosition(latestAction.to, destIndex, entries.length);
      arrow = `<svg class="map-arrow" viewBox="0 0 100 100" preserveAspectRatio="none"><line x1="${source.left}" y1="${source.top}" x2="${dest.left}" y2="${dest.top}"></line></svg>`;
    }
  }
  return `
    <section class="map-panel">
      <div class="board-header">
        <div>
          <h2>Campus Map</h2>
          <span>Decision routes are drawn between SRM blocks using the same backend step and timestamp.</span>
        </div>
      </div>
      <div class="campus-map">${arrow}${nodes}<div class="campus-road road-a"></div><div class="campus-road road-b"></div></div>
    </section>
  `;
}

function buildDecisionPanel(state) {
  const action = state.latest_decision || {};
  const latestAction = state.actions?.length ? state.actions[state.actions.length - 1] : null;
  const actionText = action.action === "redirect"
    ? `REDIRECT ${action.vehicles || latestAction?.vehicles || 0}`
    : "MONITOR";
  const route = action.action === "redirect"
    ? `${action.from || latestAction?.from || "-"} -> ${action.to || latestAction?.to || "-"}`
    : "No active route transfer";
  const llm = state.llm || {};
  const llmText = llm.used ? "LLM adjusted decision" : (llm.requested ? "LLM requested; local execution used" : "Local agent reasoning");
  return `
    <section class="decision-panel ${action.action === "redirect" ? "flash-decision" : ""}">
      <div>
        <div class="eyebrow">Current Decision</div>
        <h2>${actionText}</h2>
        <p>${route}</p>
      </div>
      <div class="decision-detail">
        <strong>${state.agent_thought || state.decision_reason || "Planner is monitoring live parking state."}</strong>
        <span>${state.decision_reason || "Waiting for the next agent decision."}</span>
        <small>${llmText}${llm.summary ? ` - ${llm.summary}` : ""}</small>
      </div>
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

function buildLearningPanel(state) {
  const learning = state.learning || {};
  const rules = learning.llm_memory_rules || [];
  const blocked = learning.blocked_routes || [];
  return `
    <section class="learning-panel">
      <div class="board-header">
        <div>
          <h2>Learning Overlay</h2>
          <span>${learning.latest_learning_insight || "Learning state will update as rewards and route outcomes are recorded."}</span>
        </div>
      </div>
      <div class="learning-grid">
        <div><span>Reward Trend</span><strong>${Number(learning.recent_reward_avg || 0).toFixed(2)}</strong></div>
        <div><span>Blocked Routes</span><strong>${blocked.length}</strong></div>
        <div><span>LLM Memory Rules</span><strong>${rules.length}</strong></div>
      </div>
      ${(blocked.length || rules.length) ? `<div class="route-tags">
        ${blocked.slice(0, 4).map((route) => `<span class="route-tag bad">Inefficient ${route}</span>`).join("")}
        ${rules.slice(0, 4).map((rule) => `<span class="route-tag good">Optimized ${rule.route_key || `${rule.from || "-"}->${rule.to || "-"}`}</span>`).join("")}
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
  return {
    left: block.x + 28 + vehicle.position.x * (block.width - 56),
    top: block.y + 124 + vehicle.position.y * (block.height - 166),
  };
}

function pickVisibleVehicles(vehicles) {
  const byBlock = new Map();
  vehicles.forEach((vehicle) => {
    const bucket = byBlock.get(vehicle.block) || [];
    bucket.push(vehicle);
    byBlock.set(vehicle.block, bucket);
  });
  const visible = [];
  byBlock.forEach((items) => {
    const sorted = [...items].sort((a, b) => (a.slot || 0) - (b.slot || 0));
    const limit = 14;
    if (sorted.length <= limit) {
      visible.push(...sorted);
      return;
    }
    const stride = sorted.length / limit;
    for (let index = 0; index < limit; index += 1) {
      visible.push(sorted[Math.floor(index * stride)]);
    }
  });
  return visible;
}

function buildArrowMarkup(action, layout) {
  if (!action || action.type !== "redirect" || !layout[action.from] || !layout[action.to]) return "";
  const source = layout[action.from];
  const destination = layout[action.to];
  const x1 = source.x + source.width / 2;
  const y1 = source.y + source.height / 2;
  const x2 = destination.x + destination.width / 2;
  const y2 = destination.y + destination.height / 2;
  return `
    <svg class="flow-overlay" viewBox="0 0 1800 1400" preserveAspectRatio="none">
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
  `;
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

function buildPreviewSlots(blockName, block, vehicles) {
  const previewSlots = 24;
  const occupied = Number(block.occupied || 0);
  const capacity = Number(block.capacity || 0);
  const occupiedPreview = capacity ? Math.min(previewSlots, Math.round((occupied / capacity) * previewSlots)) : 0;
  const vehiclesHere = vehicles.filter((vehicle) => vehicle.block === blockName).slice(0, occupiedPreview);
  return Array.from({ length: previewSlots }, (_, index) => {
    const filled = index < occupiedPreview;
    const vehicle = vehiclesHere[index];
    const icon = vehicle ? (vehicle.type === "bike" ? "🏍" : "🚗") : "";
    return `<div class="slot ${filled ? "occupied" : "free"}">${icon}</div>`;
  }).join("");
}

function buildDetailSlots(blockName, block, vehicles) {
  const capacity = Number(block.capacity || 0);
  const occupiedSet = new Map(vehicles.filter((vehicle) => vehicle.block === blockName).map((vehicle) => [vehicle.slot, vehicle]));
  const carSlots = Number(block.car_slots || capacity);
  return Array.from({ length: capacity }, (_, index) => {
    const slotNumber = index + 1;
    const vehicle = occupiedSet.get(slotNumber);
    const slotType = slotNumber <= carSlots ? "car-slot" : "bike-slot";
    const icon = vehicle ? (vehicle.type === "bike" ? "🏍" : "🚗") : "";
    const label = vehicle ? `${vehicle.type} parked` : "Free";
    return `
      <div class="detail-slot ${slotType} ${vehicle ? "filled" : "empty"}" title="Slot ${slotNumber}: ${label}">
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
        <div><span>Cars</span><strong>${formatNumber(cars)}</strong></div>
        <div><span>Bikes</span><strong>${formatNumber(bikes)}</strong></div>
      </div>
      <div class="detail-legend">
        <div><span class="legend-dot car"></span>Car slot</div>
        <div><span class="legend-dot bike"></span>Bike slot</div>
        <div><span class="legend-dot free"></span>Free slot</div>
      </div>
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
  const latestAction = state.actions?.length ? state.actions[state.actions.length - 1] : null;
  const visibleVehicles = pickVisibleVehicles(state.vehicles || []);
  const totalCapacity = blockEntries.reduce((sum, [, block]) => sum + Number(block.capacity || 0), 0);
  const totalOccupied = blockEntries.reduce((sum, [, block]) => sum + Number(block.occupied || 0), 0);
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
      ${buildDecisionPanel(state)}
      ${buildAlertsPanel(state)}
      ${buildMiniMap(state)}

      <section class="summary-strip">
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
        <div class="summary-card">
          <span>Rendered Vehicles</span>
          <strong>${visibleVehicles.length}</strong>
          <small>Visible on block previews</small>
        </div>
      </section>

      <section class="board-card">
        <div class="board-header">
          <div>
            <h2>SRM Parking Blocks</h2>
            <span>Red blocks are high occupancy, yellow are medium, green are low. Click to inspect real slots.</span>
          </div>
          <div class="legend">
            <div class="legend-item"><span class="legend-dot high"></span>High occupancy</div>
            <div class="legend-item"><span class="legend-dot medium"></span>Medium occupancy</div>
            <div class="legend-item"><span class="legend-dot low"></span>Low occupancy</div>
          </div>
        </div>

        <div class="board" style="height:${boardHeight}px">
          ${buildArrowMarkup(latestAction, layout)}
          ${blockEntries.map(([name, block]) => {
            const blockLayout = layout[name];
            const occupied = Number(block.occupied || 0);
            const capacity = Number(block.capacity || 0);
            const freeSlots = Math.max(0, capacity - occupied);
            const level = occupancyLevel(occupied, capacity);
            const activeRedirect = latestAction && (latestAction.from === name || latestAction.to === name) ? " active-block" : "";
            const sourceClass = latestAction?.from === name ? " source-block" : "";
            const destinationClass = latestAction?.to === name ? " destination-block" : "";
            const selected = selectedBlock === name ? " selected-block" : "";
            const progress = capacity ? Math.round((occupied / capacity) * 100) : 0;
            const carSlots = Number(block.car_slots || capacity);
            const bikeSlots = Number(block.bike_slots || Math.max(0, capacity - carSlots));
            return `
              <button class="block-card ${level}${activeRedirect}${sourceClass}${destinationClass}${selected}" data-block="${name}" style="transform:translate(${blockLayout?.x || 0}px, ${blockLayout?.y || 0}px); width:${blockLayout?.width || 328}px; height:${blockLayout?.height || 286}px;">
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
                  <span>Cars ${formatNumber(carSlots)}</span>
                  <span>Bikes ${formatNumber(bikeSlots)}</span>
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
      </section>

      ${buildDetailPanel(state)}
      ${buildLearningPanel(state)}

      <section class="action-card">
        <div class="board-header">
          <div>
            <h2>Redirect Timeline</h2>
            <span>Timeline entries are linked to block highlighting and movement overlays.</span>
          </div>
        </div>
        ${(state.actions || []).length ? `
          <ul class="action-list">
            ${state.actions.slice().reverse().map((action) => `
              <li class="${latestAction && latestAction.step === action.step ? "current" : ""}">
                <div class="action-step">${formatTimestamp(action.timestamp)} · Step ${action.step}</div>
                <div class="action-route">${String(action.type).toUpperCase()} ${action.vehicles} from ${action.from} to ${action.to}</div>
                <div class="action-meta">Cars ${action.car_vehicles || 0} · Bikes ${action.bike_vehicles || 0} · ${action.reason || "Agent route execution"}</div>
              </li>
            `).join("")}
          </ul>
        ` : `
          <div class="fallback-panel compact">
            <span>No redirect actions yet. The next backend redirect will animate across the source and destination blocks.</span>
          </div>
        `}
      </section>
    </div>
  `;

  animateRedirect(latestAction, previousState, state, layout);
  window.requestAnimationFrame(() => window.scrollTo({ top: scrollTop, behavior: "auto" }));
}

function animateRedirect(action, oldState, currentState, layout) {
  if (!action || action.type !== "redirect" || action.step === lastAnimatedStep) return;
  lastAnimatedStep = action.step;
  const board = document.querySelector(".board");
  if (!board || !layout[action.from] || !layout[action.to]) return;

  const previousVehicles = new Map((oldState.vehicles || []).map((vehicle) => [vehicle.id, vehicle]));
  const currentVehicles = new Map((currentState.vehicles || []).map((vehicle) => [vehicle.id, vehicle]));
  const movingIds = (action.vehicle_ids || []).slice(0, 3);
  const count = movingIds.length || Math.min(3, Number(action.vehicles || 0));
  if (!count) return;

  Array.from({ length: count }, (_, index) => movingIds[index]).forEach((vehicleId, index) => {
    const fromVehicle = previousVehicles.get(vehicleId);
    const toVehicle = currentVehicles.get(vehicleId);
    const fromPosition = fromVehicle ? vehicleScreenPosition(fromVehicle, layout) : {
      left: layout[action.from].x + layout[action.from].width / 2,
      top: layout[action.from].y + layout[action.from].height / 2,
    };
    const toPosition = toVehicle ? vehicleScreenPosition(toVehicle, layout) : {
      left: layout[action.to].x + layout[action.to].width / 2,
      top: layout[action.to].y + layout[action.to].height / 2,
    };
    const ghost = document.createElement("div");
    ghost.className = "vehicle-ghost";
    const type = toVehicle?.type || fromVehicle?.type || (index < Number(action.car_vehicles || 0) ? "car" : "bike");
    ghost.textContent = type === "bike" ? "🏍" : "🚗";
    ghost.style.left = `${fromPosition.left + index * 8}px`;
    ghost.style.top = `${fromPosition.top + index * 8}px`;
    board.appendChild(ghost);
    requestAnimationFrame(() => {
      ghost.style.left = `${toPosition.left + index * 8}px`;
      ghost.style.top = `${toPosition.top + index * 8}px`;
      ghost.style.opacity = "0.15";
    });
    window.setTimeout(() => ghost.remove(), 1300);
  });
}

async function fetchState() {
  try {
    const response = await fetch(`${API_BASE}/state?_ts=${Date.now()}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const payload = await response.json();
    const shared = payload.current_state || payload || {};
    previousState = latestState;
    const nextState = {
      step: shared.step || 0,
      updated_at: shared.updated_at || "",
      blocks: shared.blocks || {},
      vehicles: shared.vehicles || [],
      actions: shared.actions || [],
      movement_log: shared.movement_log || payload.movement_log || [],
      alerts: shared.alerts || payload.alerts || [],
      latest_decision: shared.latest_decision || payload.latest_decision || {},
      decision_reason: shared.decision_reason || payload.decision_reason || "",
      agent_thought: shared.agent_thought || payload.agent_thought || "",
      learning: shared.learning || payload.learning || {},
      llm: shared.llm || payload.llm || {},
    };
    const nextKey = JSON.stringify({
      step: nextState.step,
      updated_at: nextState.updated_at,
      vehicleCount: nextState.vehicles.length,
      actionCount: nextState.actions.length,
      alertCount: nextState.alerts.length,
      decision: nextState.latest_decision,
      selectedBlock,
    });
    latestState = nextState;
    if (nextKey !== lastRenderKey) {
      lastRenderKey = nextKey;
      render(latestState, "");
    }
  } catch (error) {
    render(latestState, error?.message || "Unable to fetch parking state.");
  }
}

app.addEventListener("click", (event) => {
  const close = event.target.closest("[data-close-detail='true']");
  if (close) {
    selectedBlock = null;
    render(latestState, "");
    return;
  }
  const blockButton = event.target.closest("[data-block]");
  if (blockButton) {
    selectedBlock = blockButton.getAttribute("data-block");
    render(latestState, "");
  }
});

window.addEventListener("resize", () => render(latestState, ""));
render(latestState, "");
fetchState();
window.setInterval(fetchState, 1000);
