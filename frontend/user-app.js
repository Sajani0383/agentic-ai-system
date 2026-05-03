const API_BASE = "http://127.0.0.1:8000";

const screen = document.getElementById("screen");
const title = document.getElementById("screen-title");
const exitForm = document.getElementById("exit-form");
const exitResult = document.getElementById("exit-result");

let snapshot = { blocks: {}, gates: {}, users: [] };
let selectedMode = "";
let returningVehicle = "";

const walkBase = {
  Gate1: { "Main Block": 3, Library: 4, "Admin Block": 5, "Hi-Tech Block": 5, "Tech Park": 8 },
  Gate2: { "Basic Eng Lab": 3, "Tech Park": 4, Library: 5, "Main Block": 7, "Admin Block": 6 },
};

function blocks() {
  return Object.keys(snapshot.blocks || {});
}

function gates() {
  return Object.keys(snapshot.gates || { Gate1: {}, Gate2: {} });
}

function blockOptions() {
  return `<option value="">Agent chooses block</option>${blocks().map((block) => `<option value="${block}">${block}</option>`).join("")}`;
}

function gateOptions() {
  return gates().map((gate) => `<option value="${gate}">${gate}</option>`).join("");
}

function buildStepRail(active = 1) {
  const steps = ["Identify", "Preferences", "Assigned"];
  return `
    <div class="step-rail" aria-label="Parking request progress">
      ${steps.map((step, index) => {
        const number = index + 1;
        const state = number < active ? "done" : number === active ? "active" : "";
        return `<div class="step-chip ${state}"><span>${number}</span><b>${step}</b></div>`;
      }).join("")}
    </div>
  `;
}

function buildVehiclePreview(type = "car") {
  return `
    <div class="vehicle-preview ${type === "bike" ? "bike" : "car"}" aria-hidden="true">
      <span class="preview-body"></span>
      <span class="preview-window"></span>
      <span class="preview-wheel front"></span>
      <span class="preview-wheel rear"></span>
    </div>
  `;
}

function blockSnapshot() {
  const ranked = Object.entries(snapshot.blocks || {})
    .map(([name, block]) => {
      const capacity = Math.max(0, Number(block.capacity || 0));
      const occupied = Math.min(capacity, Math.max(0, Number(block.occupied || 0)));
      const free = Math.max(0, capacity - occupied);
      return { name, free, capacity, ratio: capacity ? occupied / capacity : 1 };
    })
    .sort((a, b) => b.free - a.free)
    .slice(0, 3);
  if (!ranked.length) return "";
  return `
    <div class="block-snapshot">
      ${ranked.map((block) => `
        <div>
          <span>${block.name}</span>
          <strong>${block.free}</strong>
          <small>free slots</small>
        </div>
      `).join("")}
    </div>
  `;
}

function walkTime(block, gate) {
  const direct = walkBase[gate]?.[block];
  if (direct) return `${direct} min`;
  const seed = String(block || "").split("").reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return `${4 + (seed % 5)} min`;
}

function userExists(vehicleNumber) {
  const normalized = String(vehicleNumber || "").trim().toUpperCase();
  return (snapshot.users || []).some((user) => String(user.vehicle_number || "").toUpperCase() === normalized)
    || (snapshot.user_vehicles || []).some((vehicle) => String(vehicle.number || "").toUpperCase() === normalized);
}

function setTitle(text) {
  title.textContent = text;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderHome() {
  selectedMode = "";
  setTitle("Parking Assistant");
  screen.innerHTML = `
    ${buildStepRail(1)}
    <div class="assistant-intro">
      <div>
        <span>Live assistant</span>
        <strong>Tell us who is entering. The agent will assign the best available block.</strong>
      </div>
      ${buildVehiclePreview("car")}
    </div>
    <div class="choice-grid">
      <button class="choice-card" data-action="returning">
        <strong>Returning User</strong>
        <span>Use your registered vehicle number.</span>
      </button>
      <button class="choice-card" data-action="new">
        <strong>New User</strong>
        <span>Register and get a parking block.</span>
      </button>
    </div>
  `;
}

function renderReturningLookup(message = "") {
  selectedMode = "returning";
  setTitle("Returning User?");
  screen.innerHTML = `
    ${buildStepRail(1)}
    ${message ? `<div class="notice error">${message}</div>` : ""}
    <form id="returning-lookup" class="step-form">
      <label>
        Vehicle Number
        <input name="vehicle_number" autocomplete="off" placeholder="TN01AB1234" required />
      </label>
      <div class="actions">
        <button type="button" class="secondary" data-action="home">Back</button>
        <button type="submit">Continue</button>
      </div>
    </form>
  `;
}

function renderProcessing() {
  setTitle("Checking Availability");
  screen.innerHTML = `
    ${buildStepRail(3)}
    <section class="processing-card">
      <div class="loader"></div>
      <strong>Analyzing parking availability...</strong>
      <span>Checking gate flow, block capacity, and live congestion.</span>
    </section>
  `;
}

function renderReturningOptions(vehicleNumber) {
  returningVehicle = vehicleNumber.trim().toUpperCase();
  setTitle("Choose Entry Details");
  screen.innerHTML = `
    ${buildStepRail(2)}
    <div class="found-card">
      <span>Vehicle found</span>
      <strong>${returningVehicle}</strong>
    </div>
    <form id="returning-submit" class="step-form">
      <input type="hidden" name="vehicle_number" value="${returningVehicle}" />
      <label>
        Gate
        <select name="gate">${gateOptions()}</select>
      </label>
      <label>
        Preferred Block Optional
        <select name="preferred_block">${blockOptions()}</select>
      </label>
      ${blockSnapshot()}
      <div class="actions">
        <button type="button" class="secondary" data-action="returning">Back</button>
        <button type="submit">Get Parking</button>
      </div>
    </form>
  `;
}

function renderNewUser() {
  selectedMode = "new";
  setTitle("New User");
  screen.innerHTML = `
    ${buildStepRail(1)}
    <form id="new-user-submit" class="step-form">
      <label>
        Name
        <input name="name" autocomplete="name" placeholder="Your name" required />
      </label>
      <label>
        Vehicle Number
        <input name="vehicle_number" autocomplete="off" placeholder="TN01AB1234" required />
      </label>
      <div class="grid-two">
        <label>
          User Type
          <select name="user_type">
            <option value="student">Student</option>
            <option value="staff">Staff</option>
            <option value="visitor">Visitor</option>
          </select>
        </label>
        <label>
          Vehicle Type
          <select name="vehicle_type">
            <option value="car">Car</option>
            <option value="bike">Bike</option>
          </select>
        </label>
      </div>
      <label>
        Destination Optional
        <input name="destination" placeholder="Library, admin office, exam hall..." />
      </label>
      <label>
        Gate
        <select name="gate">${gateOptions()}</select>
      </label>
      ${blockSnapshot()}
      <div class="actions">
        <button type="button" class="secondary" data-action="home">Back</button>
        <button type="submit">Get Parking</button>
      </div>
    </form>
  `;
}

function renderResult(result) {
  const redirected = result.status === "redirected";
  const block = result.assigned_block || result.vehicle?.block || "-";
  const gate = result.gate || result.vehicle?.gate || "-";
  const vehicleType = result.vehicle?.type || "car";
  const vehicleNumber = result.vehicle?.number || "";
  const reason = redirected
    ? `Selected block full. Redirected to ${block}.`
    : result.reason || "Assigned due to low congestion and available capacity.";
  setTitle(redirected ? "Selected Block Full" : "Parking Assigned");
  screen.innerHTML = `
    ${buildStepRail(3)}
    <section class="result-card ${redirected ? "redirected" : ""}">
      <div class="result-topline">
        <div class="status-badge ${redirected ? "warn" : "ok"}">${redirected ? "Redirected" : "Assigned"}</div>
        ${buildVehiclePreview(vehicleType)}
      </div>
      <span>${redirected ? "Redirected to" : "Parking Assigned"}</span>
      <strong>${block}</strong>
      <div class="result-grid">
        <div><span>Gate</span><b>${gate}</b></div>
        <div><span>Walk Time</span><b>${walkTime(block, gate)}</b></div>
        <div><span>Vehicle</span><b>${vehicleNumber || "-"}</b></div>
      </div>
      <div class="route-preview">
        <span>${gate}</span>
        <i></i>
        <span>${block}</span>
      </div>
      <p>${reason}</p>
    </section>
    <div class="actions single">
      <button type="button" data-action="home">Done</button>
    </div>
  `;
}

async function submitEntry(payload) {
  renderProcessing();
  const [response] = await Promise.all([
    fetch(`${API_BASE}/user-entry`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
    delay(900),
  ]);
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Entry failed");
  await loadOptions();
  renderResult(result);
}

async function loadOptions() {
  const response = await fetch(`${API_BASE}/client-state?_ts=${Date.now()}`, { cache: "no-store" });
  const data = await response.json();
  snapshot = data.current_state || data || snapshot;
}

screen.addEventListener("click", (event) => {
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (!action) return;
  if (action === "home") renderHome();
  if (action === "returning") renderReturningLookup();
  if (action === "new") renderNewUser();
});

screen.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const payload = Object.fromEntries(new FormData(form).entries());
  try {
    if (form.id === "returning-lookup") {
      renderProcessing();
      await delay(600);
      if (!userExists(payload.vehicle_number)) {
        renderReturningLookup("Vehicle not found. Please register as a new user.");
        return;
      }
      renderReturningOptions(payload.vehicle_number);
      return;
    }
    await submitEntry(payload);
  } catch (error) {
    screen.insertAdjacentHTML("afterbegin", `<div class="notice error">${error.message || "Request failed."}</div>`);
  }
});

exitForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(exitForm).entries());
  try {
    const response = await fetch(`${API_BASE}/user-exit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Exit failed");
    if (result.status === "not_found") {
      exitResult.textContent = "Vehicle not found";
      exitResult.className = "mini-result error";
      return;
    }
    exitResult.innerHTML = `<strong>${result.notification || "Exit completed"}</strong><span>Gate: ${result.gate || "-"}</span>`;
    exitResult.className = "mini-result";
    exitForm.reset();
    await loadOptions();
  } catch (error) {
    exitResult.textContent = error.message || "Exit could not be completed.";
    exitResult.className = "mini-result error";
  }
});

loadOptions()
  .then(renderHome)
  .catch(() => {
    snapshot = { blocks: {}, gates: { Gate1: {}, Gate2: {} }, users: [] };
    renderHome();
  });
