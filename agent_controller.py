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
from llm_reasoning import get_llm_decision, summarize_state
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
        event_context = self.environment.get_event_context()
        demand = self.demand_agent.predict(state)
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
        policy_action = self.policy_agent.decide(state, demand, insight) or {"action": "none"}

        action = execution_output.get("final_action", {"action": "none"})
        mode = "agentic_loop" if action.get("action") == "redirect" else "goal_hold"

        new_state, environment_reward = self.environment.step(action)
        transition = self.environment.get_last_transition()
        kpis = transition.get("kpis", {})
        notifications = transition.get("notifications", [])
        reward_score = self.reward_agent.evaluate(state, new_state)
        self.policy_agent.update(state, action, reward_score, new_state)
        self.memory.set_q_table(self.policy_agent.export_q_table())
        summary = summarize_state(new_state)
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
            "notifications": notifications,
            "kpis": kpis,
            "reward": {
                "environment_reward": environment_reward,
                "reward_score": reward_score,
            },
        }
        self.memory.log_cycle(cycle_record)

        agent_interactions = [
            {
                "agent": "MonitoringAgent",
                "message": "Observed the live parking state from the environment.",
                "payload": state,
            },
            {
                "agent": "DemandAgent",
                "message": "Estimated zone demand pressure from occupancy and flow.",
                "payload": demand,
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

        result = {
            "mode": mode,
            "action": action or {"action": "none"},
            "policy_action": policy_action,
            "planner_output": planner_output,
            "critic_output": critic_output,
            "execution_output": execution_output,
            "goal": self.memory.get_active_goal(),
            "strategy": planner_output.get("strategy", "Balanced utilisation"),
            "event_context": transition.get("event_context", event_context),
            "notifications": notifications,
            "kpis": kpis,
            "demand": demand,
            "insight": insight,
            "environment_reward": environment_reward,
            "reward_score": reward_score,
            "state": new_state,
            "metrics": self.memory.get_metrics(),
            "summary": summary,
            "reasoning": get_llm_decision(new_state),
            "transition": transition,
            "agent_interactions": agent_interactions,
            "step_number": transition.get("step", len(self.memory.history)),
        }

        if self.logger:
            self.logger.log_step(result)

        return result
