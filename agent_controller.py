from agent_memory import AgentMemory
from agents.bayesian_agent import BayesianAgent
from agents.demand_agent import DemandAgent
from agents.monitoring_agent import MonitoringAgent
from agents.policy_agent import PolicyAgent
from agents.reward_agent import RewardAgent
from environment.parking_environment import ParkingEnvironment
from llm_reasoning import ask_llm_for_json_decision, get_llm_decision, summarize_state
from logs.logger import SimulationLogger


class AgentController:
    def __init__(self, environment=None, use_logger=False):
        self.environment = environment or ParkingEnvironment()
        self.memory = AgentMemory()
        self.monitoring_agent = MonitoringAgent()
        self.demand_agent = DemandAgent()
        self.bayesian_agent = BayesianAgent()
        self.policy_agent = PolicyAgent(self.environment.zones)
        self.reward_agent = RewardAgent()
        self.logger = SimulationLogger() if use_logger else None

    def reset(self):
        state = self.environment.reset()
        self.memory = AgentMemory()
        if self.logger:
            self.logger.reset_logs()
        return state

    def step(self):
        state = self.monitoring_agent.observe(self.environment)
        demand = self.demand_agent.predict(state)
        insight = self.bayesian_agent.infer(state)

        llm_action = ask_llm_for_json_decision(
            state,
            demand,
            insight,
            self.memory.get_metrics(),
        )

        if llm_action.get("action") == "redirect":
            action = llm_action
            mode = "llm"
        else:
            action = self.policy_agent.decide(state, demand, insight)
            mode = "policy"

        new_state, environment_reward = self.environment.step(action)
        reward_score = self.reward_agent.evaluate(state, new_state)
        self.memory.add(new_state)
        summary = summarize_state(new_state)

        result = {
            "mode": mode,
            "action": action or {"action": "none"},
            "llm_action": llm_action,
            "demand": demand,
            "insight": insight,
            "environment_reward": environment_reward,
            "reward_score": reward_score,
            "state": new_state,
            "metrics": self.memory.get_metrics(),
            "summary": summary,
            "reasoning": get_llm_decision(new_state),
        }

        if self.logger:
            self.logger.log_step(result)

        return result
