from agent_memory import AgentMemory
from agents.bayesian_agent import BayesianAgent
from agents.critic_agent import CriticAgent
from agents.demand_agent import DemandAgent
from agents.executor_agent import ExecutorAgent
from agents.monitoring_agent import MonitoringAgent
from agents.planner_agent import PlannerAgent
from agents.policy_agent import PolicyAgent
from agents.reward_agent import RewardAgent
from environment.parking_environment import ParkingEnvironment
from llm_reasoning import get_operational_reasoning, summarize_state
from logs.logger import SimulationLogger
from tools import build_runtime_tools


class AgentController:
    def __init__(self, environment=None, memory=None, use_logger=False):
        self.environment = environment or ParkingEnvironment()
        self.memory = memory or AgentMemory()
        self.monitoring_agent = MonitoringAgent()
        self.demand_agent = DemandAgent()
        self.bayesian_agent = BayesianAgent()
        self.planner_agent = PlannerAgent()
        self.critic_agent = CriticAgent()
        self.executor_agent = ExecutorAgent()
        self.policy_agent = PolicyAgent(self.environment.zones)
        self.policy_agent.load_q_table(self.memory.get_q_table())
        self.reward_agent = RewardAgent()
        self.logger = SimulationLogger() if use_logger else None

    def reset(self, clear_memory=False):
        state = self.environment.reset()
        if clear_memory:
            self.memory.reset()
        if self.logger:
            self.logger.reset_logs()
        return state

    def step(self):
        state = self.monitoring_agent.observe(self.environment)
        monitoring_report = self.monitoring_agent.get_last_observation()
        event_context = self.environment.get_event_context()
        operational_signals = self.environment.get_operational_signals()
        demand = self.demand_agent.predict(
            state,
            event_context=event_context,
            operational_signals=operational_signals,
            simulated_hour=getattr(self.environment, "simulated_hour", None),
            historical_states=self.environment.get_trend(),
        )
        demand_report = self.demand_agent.get_last_report()
        insight = self.bayesian_agent.infer(state)
        memory_metrics = self.memory.get_metrics()
        tools = build_runtime_tools(self.environment, self.memory)

        planner_output = self.planner_agent.plan(
            state,
            demand,
            insight,
            memory_metrics,
            tools,
        )
        goal = planner_output.get("goal", {})
        current_goal = self.memory.get_active_goal()
        if goal and goal != current_goal:
            self.memory.set_goal(goal)

        critic_output = self.critic_agent.review(
            planner_output,
            state,
            demand,
            insight,
            tools,
        )
        execution_output = self.executor_agent.execute(critic_output, self.environment)
        policy_action = self.policy_agent.decide(
            state,
            demand,
            insight,
            event_context=event_context,
        ) or {"action": "none"}

        action = execution_output.get("final_action", {"action": "none"})
        mode = "agentic_loop" if action.get("action") == "redirect" else "goal_hold"

        new_state, environment_reward = self.environment.step(action)
        transition = self.environment.get_last_transition()
        kpis = transition.get("kpis", {})
        notifications = transition.get("notifications", [])
        reward_score = self.reward_agent.evaluate(
            state,
            new_state,
            action=action,
            demand=demand,
            event_context=transition.get("event_context", event_context),
            kpis=kpis,
            transition=transition,
        )
        self.demand_agent.update_from_feedback(demand, kpis=kpis)
        self.policy_agent.update(
            state,
            action,
            reward_score,
            new_state,
            demand=demand,
            insight=insight,
            execution_feedback=execution_output,
        )
        self.memory.set_q_table(self.policy_agent.export_q_table())
        summary = summarize_state(new_state)
        replan_triggered = self._should_replan(kpis, goal=self.memory.get_active_goal())
        autonomy = self._build_autonomy_status(
            transition=transition,
            planner_output=planner_output,
            critic_output=critic_output,
            replan_triggered=replan_triggered,
        )
        self.memory.update_learning_signal(
            self.environment.get_scenario_mode(),
            action,
            reward_score,
            kpis=kpis,
        )

        self.memory.add(
            new_state,
            transition=transition,
            summary=summary,
            step=transition.get("step"),
            kpis=kpis,
            notifications=notifications,
            event_context=transition.get("event_context", event_context),
        )
        cycle_record = {
            "step": transition.get("step"),
            "goal": self.memory.get_active_goal(),
            "planner_output": planner_output,
            "critic_output": critic_output,
            "execution_output": execution_output,
            "policy_baseline": policy_action,
            "event_context": transition.get("event_context", event_context),
            "operational_signals": transition.get("dynamic_signals", operational_signals),
            "notifications": notifications,
            "kpis": kpis,
            "autonomy": autonomy,
            "demand_report": demand_report,
            "reward": {
                "environment_reward": environment_reward,
                "reward_score": reward_score,
            },
        }
        self.memory.log_cycle(cycle_record)

        agent_interactions = [
            {
                "agent": "MonitoringAgent",
                "message": "Observed, validated, and normalized the live parking state from the environment.",
                "payload": monitoring_report,
            },
            {
                "agent": "DemandAgent",
                "message": "Forecasted demand using flow, scarcity, trend, time, event context, and uncertainty.",
                "payload": demand_report,
            },
            {
                "agent": "EventContext",
                "message": "Loaded the active campus event and the current allocation strategy.",
                "payload": transition.get("event_context", event_context),
            },
            {
                "agent": "BayesianAgent",
                "message": "Ranked congestion spread and confidence.",
                "payload": insight,
            },
            {
                "agent": "PlannerAgent",
                "message": "Created a multi-step goal and proposed an action using runtime tools.",
                "payload": planner_output,
            },
            {
                "agent": "CriticAgent",
                "message": "Stress-tested the proposed action for safety and goal alignment.",
                "payload": critic_output,
            },
            {
                "agent": "ExecutorAgent",
                "message": "Prepared the final executable action for this step.",
                "payload": execution_output,
            },
            {
                "agent": "PolicyBaseline",
                "message": "Computed the deterministic baseline used for comparison.",
                "payload": policy_action,
            },
            {
                "agent": "RewardAgent",
                "message": "Measured the impact of the executed transition.",
                "payload": {
                    "environment_reward": environment_reward,
                    "reward_score": reward_score,
                },
            },
        ]
        reasoning = get_operational_reasoning(new_state)

        result = {
            "mode": "replan_loop" if replan_triggered else mode,
            "action": action or {"action": "none"},
            "policy_action": policy_action,
            "planner_output": planner_output,
            "critic_output": critic_output,
            "execution_output": execution_output,
            "goal": self.memory.get_active_goal(),
            "strategy": planner_output.get("strategy", "Balanced utilisation"),
            "event_context": transition.get("event_context", event_context),
            "operational_signals": transition.get("dynamic_signals", operational_signals),
            "notifications": notifications,
            "kpis": kpis,
            "demand": demand,
            "demand_report": demand_report,
            "insight": insight,
            "environment_reward": environment_reward,
            "reward_score": reward_score,
            "autonomy": autonomy,
            "state": new_state,
            "monitoring_report": monitoring_report,
            "metrics": self.memory.get_metrics(),
            "summary": summary,
            "reasoning": reasoning["text"],
            "reasoning_source": reasoning["source"],
            "transition": transition,
            "agent_interactions": agent_interactions,
            "step_number": transition.get("step", len(self.memory.history)),
        }

        if self.logger:
            self.logger.log_step(result)

        return result

    def _should_replan(self, kpis, goal):
        target_search_time = goal.get("target_search_time_min", 4.0) if goal else 4.0
        return bool(
            kpis.get("estimated_search_time_min", 0) > target_search_time
            or kpis.get("queue_length", 0) >= 4
            or kpis.get("resilience_score", 100) < 60
        )

    def _build_autonomy_status(self, transition, planner_output, critic_output, replan_triggered):
        kpis = transition.get("kpis", {})
        signals = transition.get("dynamic_signals", {})
        return {
            "replan_triggered": replan_triggered,
            "projection_horizon_steps": planner_output.get("goal", {}).get("horizon_steps", 0),
            "blocked_zone": signals.get("blocked_zone"),
            "queue_length": signals.get("queue_length", 0),
            "resilience_score": kpis.get("resilience_score", 0),
            "critic_risk": critic_output.get("risk_level", "low"),
        }
