import { useEffect, useMemo, useRef, useState } from "react";

const EMPTY_STATE = {
  step: 0,
  updated_at: "",
  blocks: {},
  vehicles: [],
  actions: []
};

function buildBlockLayout(blockEntries, containerWidth) {
  const gap = 18;
  const blockWidth = 220;
  const blockHeight = 156;
  const columns = Math.max(1, Math.floor((containerWidth + gap) / (blockWidth + gap)));
  return blockEntries.reduce((acc, [name], index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    acc[name] = {
      x: column * (blockWidth + gap),
      y: row * (blockHeight + gap),
      width: blockWidth,
      height: blockHeight
    };
    return acc;
  }, {});
}

function vehicleScreenPosition(vehicle, layout) {
  const block = layout[vehicle.block];
  if (!block) {
    return { left: 0, top: 0 };
  }
  const x = block.x + 18 + vehicle.position.x * (block.width - 36);
  const y = block.y + 18 + vehicle.position.y * (block.height - 42);
  return { left: x, top: y };
}

export default function App() {
  const [state, setState] = useState(EMPTY_STATE);
  const [error, setError] = useState("");
  const [containerWidth, setContainerWidth] = useState(980);
  const boardRef = useRef(null);

  useEffect(() => {
    const fetchState = async () => {
      try {
        const response = await fetch(`/state?_ts=${Date.now()}`, {
          cache: "no-store",
          headers: { Accept: "application/json" }
        });
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        const payload = await response.json();
        const shared = payload.current_state || payload;
        setState({
          step: shared.step || 0,
          updated_at: shared.updated_at || "",
          blocks: shared.blocks || {},
          vehicles: shared.vehicles || [],
          actions: shared.actions || []
        });
        setError("");
      } catch (err) {
        setError(err.message || "Unable to fetch parking state.");
      }
    };

    fetchState();
    const intervalId = window.setInterval(fetchState, 1000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (!boardRef.current) {
      return undefined;
    }
    const updateWidth = () => {
      setContainerWidth(boardRef.current?.clientWidth || 980);
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(boardRef.current);
    return () => observer.disconnect();
  }, []);

  const blockEntries = useMemo(() => Object.entries(state.blocks || {}), [state.blocks]);
  const layout = useMemo(() => buildBlockLayout(blockEntries, containerWidth), [blockEntries, containerWidth]);
  const totalRows = Math.max(1, Math.ceil(blockEntries.length / Math.max(1, Math.floor((containerWidth + 18) / (220 + 18)))));
  const boardHeight = totalRows * 156 + Math.max(0, totalRows - 1) * 18;
  const latestAction = state.actions?.length ? state.actions[state.actions.length - 1] : null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">SRM Smart Parking</div>
          <h1>Live Shared-State Simulation</h1>
        </div>
        <div className="status-cluster">
          <div className="status-pill">Step {state.step}</div>
          <div className="status-pill muted">{state.updated_at || "Waiting for backend"}</div>
        </div>
      </header>

      {error ? (
        <div className="fallback-panel">
          <strong>API unavailable.</strong>
          <span>{error}</span>
        </div>
      ) : null}

      <section className="summary-strip">
        <div className="summary-card">
          <span>Blocks</span>
          <strong>{blockEntries.length}</strong>
        </div>
        <div className="summary-card">
          <span>Vehicles</span>
          <strong>{state.vehicles.length}</strong>
        </div>
        <div className="summary-card">
          <span>Latest Action</span>
          <strong>{latestAction ? `${latestAction.type.toUpperCase()} ${latestAction.vehicles}` : "NONE"}</strong>
        </div>
      </section>

      <section className="board-card">
        <div className="board-header">
          <h2>Parking Blocks</h2>
          <span>Polling `/state` every 1 second</span>
        </div>
        <div ref={boardRef} className="board" style={{ height: `${boardHeight}px` }}>
          {blockEntries.map(([name, block]) => {
            const blockLayout = layout[name];
            const occupied = Number(block.occupied || 0);
            const capacity = Number(block.capacity || 0);
            const freeSlots = Math.max(0, capacity - occupied);
            return (
              <div
                key={name}
                className="block-card"
                style={{
                  transform: `translate(${blockLayout?.x || 0}px, ${blockLayout?.y || 0}px)`,
                  width: `${blockLayout?.width || 220}px`,
                  height: `${blockLayout?.height || 156}px`
                }}
              >
                <div className="block-name">{name}</div>
                <div className="block-metrics">
                  <div><span>Occupied</span><strong>{occupied}</strong></div>
                  <div><span>Free</span><strong>{freeSlots}</strong></div>
                  <div><span>Capacity</span><strong>{capacity}</strong></div>
                </div>
                <div className="block-progress">
                  <div style={{ width: `${capacity ? (occupied / capacity) * 100 : 0}%` }} />
                </div>
              </div>
            );
          })}

          <div className="vehicle-layer">
            {state.vehicles.map((vehicle) => {
              const screen = vehicleScreenPosition(vehicle, layout);
              return (
                <div
                  key={vehicle.id}
                  className={`vehicle ${vehicle.type}`}
                  style={{ left: `${screen.left}px`, top: `${screen.top}px` }}
                  title={`${vehicle.type} ${vehicle.id} in ${vehicle.block} slot ${vehicle.slot}`}
                >
                  {vehicle.type === "bike" ? "B" : "C"}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="action-card">
        <div className="board-header">
          <h2>Redirect Activity</h2>
          <span>Vehicles animate from source to destination using shared backend IDs</span>
        </div>
        {state.actions.length ? (
          <ul className="action-list">
            {state.actions.slice().reverse().map((action) => (
              <li key={`${action.step}-${action.from}-${action.to}`}>
                <strong>Step {action.step}</strong>
                <span>{action.type.toUpperCase()} {action.vehicles} from {action.from} to {action.to}</span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="fallback-panel compact">
            <span>No redirect actions yet. The frontend will animate the next backend redirect automatically.</span>
          </div>
        )}
      </section>
    </div>
  );
}
