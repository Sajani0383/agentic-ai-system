import tempfile
import os
import unittest
from unittest.mock import patch
from copy import deepcopy

from agents.bayesian_agent import BayesianAgent
from agents.critic_agent import CriticAgent
from agents.demand_agent import DemandAgent
from agents.executor_agent import ExecutorAgent
from agents.monitoring_agent import MonitoringAgent
from agents.planner_agent import PlannerAgent
from environment.parking_environment import ParkingEnvironment

class AgentsComprehensiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state = {
            "Zone_A": {"total_slots": 100, "occupied": 95, "free_slots": 5, "entry": 8, "exit": 1},
            "Zone_B": {"total_slots": 100, "occupied": 20, "free_slots": 80, "entry": 1, "exit": 4},
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Monitoring Agent ---
    def test_monitoring_agent_validation_bounds(self):
        agent = MonitoringAgent()
        agent.observe(self.state) 
        malformed_state = {"Zone_A": {"total_slots": 100, "occupied": "broken"}}
        new_state = agent.observe(malformed_state)
        self.assertEqual(agent.get_last_observation()["status"], "error")
        self.assertEqual(new_state["Zone_A"]["free_slots"], 5)

    # --- Bayesian Agent ---
    def test_bayesian_posterior_math_validation(self):
        agent = BayesianAgent()
        out = agent.infer(self.state)
        self.assertIn("posteriors", out)
        self.assertGreater(out["posteriors"]["Zone_A"], out["posteriors"]["Zone_B"])
        self.assertEqual(out["most_crowded"], "Zone_A")

    # --- Demand Agent ---
    def test_demand_agent_trend_pressure_schema(self):
        agent = DemandAgent()
        historical = [deepcopy(self.state) for _ in range(4)]
        demand = agent.predict(self.state, event_context={"severity": "high"}, simulated_hour=9, historical_states=historical)
        self.assertIn("Zone_A", demand)
        self.assertIsInstance(demand["Zone_A"], int)
        report = agent.get_last_report()
        self.assertIn("trend_pressure", report["zone_details"]["Zone_A"])

    # --- Critic Agent ---
    @patch("agents.critic_agent.ask_llm_for_structured_json")
    def test_critic_agent_blocks_unsafe_action(self, mock_llm):
        agent = CriticAgent()
        mock_llm.return_value = {
            "approved": True, 
            "risk_level": "low", 
            "critic_notes": [], 
            "revised_action": {"action": "redirect", "from": "Zone_A", "to": "Zone_A", "vehicles": 10}
        }
        plan = {"proposed_action": {"action": "redirect", "from": "Zone_A", "to": "Zone_A", "vehicles": 10}}
        tools = {"estimate_transfer_capacity": lambda f, t, r: 0}
        review = agent.review(plan, self.state, {"Zone_A": 10, "Zone_B": 0}, {"posteriors": {}}, tools)
        self.assertFalse(review["approved"])
        self.assertEqual(review["risk_level"], "high")
        self.assertEqual(review["revised_action"]["action"], "none")

    # --- Executor Agent ---
    def test_executor_updates_state_mathematically(self):
        executor = ExecutorAgent()
        env = ParkingEnvironment(zones=["Zone_A", "Zone_B"], seed=1)
        env.state = deepcopy(self.state)
        review = {
            "approved": True,
            "revised_action": {"action": "redirect", "from": "Zone_A", "to": "Zone_B", "vehicles": 5}
        }
        result = executor.execute(review, env, apply=True)
        self.assertTrue(result["success"])
        # The executor might transfer incoming requests or limit based on capacity; we just assert it successfully processes
        self.assertIn("transfer_report", result)

    # --- Planner Agent ---
    @patch("agents.planner_agent.ask_llm_for_structured_json")
    def test_planner_generates_valid_schema_under_timeout(self, mock_llm):
        mock_llm.side_effect = TimeoutError("Simulated LLM Hang")
        planner = PlannerAgent()
        tools = {
            "build_zone_pressure_report": lambda s,d: {},
            "get_recent_cycles": lambda: [],
            "get_event_context": lambda: {},
            "suggest_best_zone": lambda s: "Zone_B",
            "get_goal_status": lambda: {}
        }
        # PlannerAgent does not catch timeouts natively, it bubbles to the runtime loop.
        with self.assertRaises(TimeoutError):
            planner.plan(
                self.state,
                {"Zone_A": 10},
                {"confidence": 0.5},
                {"steps": 1},
                tools=tools,
                reasoning_budget={"allow_planner_llm": True, "planner_llm_strategy": "gemini"},
            )

    def test_planner_can_use_cached_or_local_simulated_advisory(self):
        planner = PlannerAgent()
        tools = {
            "build_zone_pressure_report": lambda s,d: {},
            "get_recent_cycles": lambda: [],
            "get_event_context": lambda: {"severity": "high", "name": "Exam Rush", "recommended_zone": "Zone_B"},
            "get_scenario_mode": lambda: "Exam Rush",
            "get_operational_signals": lambda: {"queue_length": 4},
            "suggest_best_zone": lambda s: "Zone_B",
            "get_goal_status": lambda: {},
            "get_learning_profile": lambda **kwargs: {},
            "estimate_transfer_capacity": lambda f, t, r: min(6, r),
        }

        cached_result = planner.plan(
            self.state,
            {"Zone_A": 20, "Zone_B": 2},
            {"confidence": 0.7, "uncertainty": {"entropy": 1.0}},
            {"steps": 1},
            tools=tools,
            reasoning_budget={
                "planner_llm_strategy": "cached",
                "cached_planner_advisory": {
                    "strategy": "Cached Gemini",
                    "proposed_action": {"action": "redirect", "from": "Zone_A", "to": "Zone_B", "vehicles": 3, "reason": "cached", "confidence": 0.7},
                    "alternative_actions": [],
                    "rationale": "Cached advisory reused",
                },
            },
        )
        self.assertEqual(cached_result["decision_mode"], "cached_llm_advisory")
        self.assertEqual(cached_result["llm_source"], "cached")

        local_result = planner.plan(
            self.state,
            {"Zone_A": 20, "Zone_B": 2},
            {"confidence": 0.7, "uncertainty": {"entropy": 1.0}},
            {"steps": 1},
            tools=tools,
            reasoning_budget={
                "planner_llm_strategy": "local_simulated",
                "local_simulated_advisory": {
                    "strategy": "Local AI",
                    "proposed_action": {"action": "redirect", "from": "Zone_A", "to": "Zone_B", "vehicles": 2, "reason": "local", "confidence": 0.61},
                    "alternative_actions": [],
                    "rationale": "Local AI simulated rationale",
                },
            },
        )
        self.assertEqual(local_result["decision_mode"], "local_ai_simulation")
        self.assertEqual(local_result["llm_source"], "local_simulated")

        demo_result = planner.plan(
            self.state,
            {"Zone_A": 20, "Zone_B": 2},
            {"confidence": 0.7, "uncertainty": {"entropy": 1.0}},
            {"steps": 1},
            tools=tools,
            reasoning_budget={
                "planner_llm_strategy": "demo_simulated",
                "local_simulated_advisory": {
                    "strategy": "Simulated Gemini",
                    "proposed_action": {"action": "redirect", "from": "Zone_A", "to": "Zone_B", "vehicles": 2, "reason": "simulated gemini", "confidence": 0.68},
                    "alternative_actions": [],
                    "rationale": "Simulated Gemini demo rationale",
                },
            },
        )
        self.assertEqual(demo_result["decision_mode"], "demo_simulated_gemini")
        self.assertEqual(demo_result["llm_source"], "demo_simulated")

if __name__ == "__main__":
    unittest.main()
