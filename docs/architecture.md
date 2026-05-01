# SRM Smart Parking Agentic AI Architecture

```mermaid
flowchart LR
    E[ParkingEnvironment<br/>SRM blocks, events, queues] --> M[MonitoringAgent]
    M --> D[DemandAgent]
    D --> B[BayesianAgent]
    B --> P[PlannerAgent]
    P --> C[CriticAgent]
    C --> X[ExecutorAgent]
    X --> E
    X --> R[RewardAgent]
    R --> MEM[AgentMemory<br/>routes, rewards, Q-table]
    MEM --> P
    MEM --> POL[PolicyAgent<br/>advisory only]
    POL --> P
    E --> API[FastAPI Runtime]
    API --> UI[Streamlit Dashboard]
    API --> VIS[React/Vite Visualizer]
    API --> N[Mock Notification Feed]
    LLM[Optional Gemini Reasoning] -. budgeted advisory .-> P
    LLM -. safety advisory .-> C
```

## Control Invariants

- Planner proposes actions, but cannot execute directly.
- Critic validates safety, utility, risk, and capacity constraints.
- Executor validates executable vehicle count before environment application.
- PolicyAgent is advisory only and cannot override the critic-approved planner path.
- AgentMemory records successful routes, failed routes, reward trends, and Q-table updates.
- Gemini is optional and budgeted. Local reasoning keeps the system functional when keys, quota, or network are unavailable.

## Presentation View

The strongest demo path is:

1. Dashboard `SRM Operations` for live decision impact.
2. `Agent Loop` for planner/critic/executor trace.
3. `Reasoning` for explainability and LLM usage.
4. `Benchmark` for evidence against no-agent baseline.
5. `Prepare Run Report` for exportable proof.

## Real-World Extension Path

- Replace simulated entry/exit signals with gate counters, camera counts, or ANPR events.
- Replace mock notifications with mobile push, SMS, or signage APIs.
- Persist state in a database when multiple operators need shared history.
- Add admin approval policies for high-impact redirects.
- Add role-based access control before production deployment.

