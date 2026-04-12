import os
import tempfile
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import adk.agent_manager as agent_manager
from adk.agent_api import app
from adk.agent_manager import (
    AgentManagerError,
    get_memory_report,
    get_policy_learning_report,
    preprocess_user_input,
    run_agent_loop,
    set_goal,
)
from adk.trace_logger import TraceLogger
from agents.bayesian_agent import BayesianAgent
from agents.critic_agent import CriticAgent
from agents.demand_agent import DemandAgent
from agents.executor_agent import ExecutorAgent
from agents.monitoring_agent import MonitoringAgent
from agents.planner_agent import PlannerAgent
from agents.policy_agent import PolicyAgent
from agents.reward_agent import RewardAgent
from agent_controller import AgentController
from agent_memory import AgentMemory
from communication.message_bus import MessageBus
from environment.parking_environment import ParkingEnvironment
from logs.logger import SimulationLogger
from ml import predict as demand_predictor
from ml import train_model as train_model_module
from services.parking_runtime import ParkingRuntimeService


class SimulationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory_path = os.path.join(self.temp_dir.name, "agent_memory.json")
        self.runtime_path = os.path.join(self.temp_dir.name, "runtime_state.json")
        self.notification_path = os.path.join(self.temp_dir.name, "notifications.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_agent_controller_step_returns_agentic_payload(self):
        environment = ParkingEnvironment()
        memory = AgentMemory(storage_path=self.memory_path)
        controller = AgentController(environment=environment, memory=memory)

        result = controller.step()

        self.assertIn("planner_output", result)
        self.assertIn("critic_output", result)
        self.assertIn("execution_output", result)
        self.assertIn("kpis", result)
        self.assertIn("notifications", result)
        self.assertIn("agent_interactions", result)
        self.assertIn("monitoring_report", result)
        self.assertEqual(result["monitoring_report"]["status"], "ok")
        self.assertEqual(result["reasoning_source"], "local_operational_summary")
        self.assertTrue(memory.get_q_table() is not None)

    def test_monitoring_agent_normalizes_dict_input_and_logs_trace(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "monitoring_trace.json"),
        )
        agent = MonitoringAgent(logger=logger)
        raw_state = {
            "A": {"total_slots": 100, "occupied": 70, "entry": 4, "exit": 2},
            "B": {"total_slots": 100, "occupied": 20, "free_slots": 80, "entry": 1, "exit": 3},
        }

        state = agent.observe(raw_state)

        self.assertEqual(state["A"]["free_slots"], 30)
        self.assertEqual(agent.get_last_observation()["status"], "ok")
        self.assertEqual(logger.get_by_event("monitoring_observation")[0]["level"], "INFO")

    def test_monitoring_agent_handles_invalid_input_without_crashing(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "monitoring_error_trace.json"),
        )
        agent = MonitoringAgent(logger=logger)

        fallback_state = agent.observe(
            {
                "A": {"total_slots": 100, "occupied": 60, "entry": 5, "exit": 2},
            }
        )
        invalid_result = agent.observe({"A": {"total_slots": 100, "occupied": "bad", "entry": 1, "exit": 0}})

        self.assertEqual(invalid_result, fallback_state)
        self.assertEqual(agent.get_last_observation()["status"], "error")
        self.assertEqual(logger.get_by_event("monitoring_observation")[-1]["level"], "ERROR")

    def test_bayesian_agent_calculates_posteriors_and_updates_beliefs(self):
        agent = BayesianAgent()
        state = {
            "A": {"total_slots": 100, "occupied": 95, "free_slots": 5, "entry": 8, "exit": 1},
            "B": {"total_slots": 100, "occupied": 30, "free_slots": 70, "entry": 1, "exit": 4},
        }

        first = agent.infer(state)
        first_beliefs = agent.get_beliefs()
        second = agent.infer(state)

        self.assertIn("priors", first)
        self.assertIn("likelihoods", first)
        self.assertIn("posteriors", first)
        self.assertIn("uncertainty", first)
        self.assertGreater(first["posteriors"]["A"], first["posteriors"]["B"])
        self.assertEqual(first["most_crowded"], "A")
        self.assertEqual(first["best_zone"], "B")
        self.assertNotEqual(first_beliefs["A"], agent.get_beliefs()["A"])
        self.assertGreater(second["priors"]["A"], first["priors"]["A"])

    def test_critic_agent_uses_deterministic_risk_gate_and_trace_logging(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "critic_trace.json"),
        )
        critic = CriticAgent(logger=logger)
        state = {
            "A": {"total_slots": 100, "occupied": 95, "free_slots": 5, "entry": 8, "exit": 1},
            "B": {"total_slots": 100, "occupied": 99, "free_slots": 1, "entry": 4, "exit": 1},
            "C": {"total_slots": 100, "occupied": 40, "free_slots": 60, "entry": 1, "exit": 3},
        }
        demand = {"A": 20, "B": 16, "C": 2}
        insight = {
            "confidence": 0.88,
            "posteriors": {"A": 0.91, "B": 0.84, "C": 0.22},
            "uncertainty": {"entropy": 1.2},
        }
        plan = {
            "strategy": "Safety test",
            "tool_calls": [],
            "proposed_action": {
                "action": "redirect",
                "from": "A",
                "to": "B",
                "vehicles": 8,
                "reason": "test",
                "confidence": 0.9,
            },
        }
        tools = {
            "get_event_context": lambda: {
                "severity": "critical",
                "focus_zone": "A",
                "recommended_zone": "C",
                "time_window": "09:00 - 10:00",
            },
            "get_scenario_mode": lambda: "Exam Rush",
            "get_operational_signals": lambda: {"queue_length": 5, "blocked_zone": "B"},
            "get_learning_profile": lambda **_kwargs: {"route_profile": {"avg_reward": -3.0}},
            "estimate_transfer_capacity": lambda _from, _to, requested: min(1, requested),
        }
        unsafe_llm_review = {
            "approved": True,
            "risk_level": "low",
            "critic_notes": ["Unsafe LLM approval should not win."],
            "revised_action": plan["proposed_action"],
        }

        with patch("agents.critic_agent.ask_llm_for_structured_json", return_value=unsafe_llm_review):
            review = critic.review(plan, state, demand, insight, tools)

        self.assertFalse(review["approved"])
        self.assertEqual(review["risk_level"], "high")
        self.assertEqual(review["revised_action"], {"action": "none"})
        self.assertGreaterEqual(review["risk_score"], 70)
        self.assertIn("risk_probability", review)
        self.assertIn("alternative_actions", review)
        self.assertEqual(review["alternative_actions"][0]["to"], "C")
        self.assertTrue(review["replan_recommendation"]["required"])
        self.assertIn("learning_feedback", review)
        self.assertIn("critic_review", logger.get_traces()[0]["event"])

    def test_demand_agent_uses_time_event_history_and_uncertainty(self):
        agent = DemandAgent()
        historical_states = [
            {
                "A": {"total_slots": 100, "occupied": 70 + index, "free_slots": 30 - index, "entry": 5, "exit": 2},
                "B": {"total_slots": 100, "occupied": 30, "free_slots": 70, "entry": 1, "exit": 3},
            }
            for index in range(4)
        ]
        state = {
            "A": {"total_slots": 100, "occupied": 92, "free_slots": 8, "entry": 9, "exit": 1},
            "B": {"total_slots": 100, "occupied": 35, "free_slots": 65, "entry": 1, "exit": 4},
        }
        event_context = {
            "name": "Exam Rush",
            "severity": "high",
            "focus_zone": "A",
            "recommended_zone": "B",
            "zone_multipliers": {"A": 1.4, "B": 1.0},
        }

        demand = agent.predict(
            state,
            event_context=event_context,
            operational_signals={"queue_length": 5, "weather": "Rain Surge"},
            simulated_hour=9,
            historical_states=historical_states,
        )
        report = agent.get_last_report()

        self.assertIsInstance(demand["A"], int)
        self.assertGreater(demand["A"], demand["B"])
        self.assertEqual(report["time_context"]["time_bucket"], "morning_arrival")
        self.assertIn("confidence", report)
        self.assertIn("uncertainty", report)
        self.assertIn("zone_details", report)
        self.assertGreater(report["zone_details"]["A"]["trend_pressure"], 0.5)

    def test_executor_agent_handles_partial_and_direct_execution(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "executor_trace.json"),
        )
        executor = ExecutorAgent(logger=logger)
        environment = ParkingEnvironment(
            zones=["A", "B"],
            seed=1,
        )
        environment.state = {
            "A": {"total_slots": 100, "occupied": 92, "entry": 6, "exit": 0},
            "B": {"total_slots": 100, "occupied": 96, "entry": 0, "exit": 0},
        }
        review = {
            "approved": True,
            "risk_level": "low",
            "risk_score": 12,
            "revised_action": {
                "action": "redirect",
                "from": "A",
                "to": "B",
                "vehicles": 10,
            },
        }

        prepared = executor.execute(review, environment, apply=False)
        self.assertTrue(prepared["success"])
        self.assertFalse(prepared["applied"])
        self.assertTrue(prepared["partial_execution"])
        self.assertEqual(prepared["final_action"]["vehicles"], 4)
        self.assertEqual(environment.get_state()["A"]["occupied"], 92)

        applied = executor.execute(review, environment, apply=True)
        self.assertTrue(applied["success"])
        self.assertTrue(applied["applied"])
        self.assertEqual(applied["executed_vehicles"], 4)
        self.assertEqual(applied["transfer_report"]["mode"], "incoming_reroute")
        self.assertEqual(environment.get_state()["A"]["occupied"], 92)
        self.assertEqual(environment.get_state()["B"]["occupied"], 96)
        self.assertIn("executor_execution", logger.get_traces()[0]["event"])

    def test_environment_redirect_reroutes_arrivals_instead_of_moving_parked_cars(self):
        environment = ParkingEnvironment(zones=["A", "B"], seed=1)
        environment.state = {
            "A": {"total_slots": 100, "occupied": 92, "entry": 0, "exit": 0},
            "B": {"total_slots": 100, "occupied": 40, "entry": 0, "exit": 0},
        }
        environment.rng = SimpleNamespace(uniform=lambda _a, _b: 0.2)
        environment._advance_time = lambda: None
        environment._build_dynamic_signals = lambda _event_context: {}
        environment._time_multiplier = lambda _zone: 1.0
        environment._dynamic_signal_multiplier = lambda _zone, _signals, _event_context: 1.0
        environment.get_event_context = lambda: {
            "name": "Exam Rush",
            "severity": "high",
            "description": "test",
            "focus_zone": "A",
            "recommended_zone": "B",
            "allocation_strategy": "Demand smoothing",
            "zone_multipliers": {"A": 1.0, "B": 1.0},
            "user_advisory": "test",
            "time_window": "09:00 - 10:00",
        }

        with patch("environment.parking_environment.predict_demand", return_value=100):
            before = environment.get_state()
            after, _reward = environment.step({"action": "redirect", "from": "A", "to": "B", "vehicles": 5})

        self.assertEqual(environment.get_last_transition()["transfer_detail"]["moved"], 5)
        self.assertLessEqual(after["A"]["occupied"], before["A"]["occupied"])
        self.assertGreaterEqual(after["B"]["occupied"], before["B"]["occupied"])
        self.assertEqual(before["A"]["occupied"], 92)

    def test_environment_exposes_modular_summary_and_config(self):
        environment = ParkingEnvironment(seed=5, config={"history_limit": 20, "queue_warning_threshold": 3})

        summary = environment.get_environment_summary()

        self.assertEqual(summary["environment_type"], "dynamic event-driven stochastic parking simulation")
        self.assertEqual(summary["config"]["history_limit"], 20)
        self.assertEqual(len(summary["step_breakdown"]), 5)

    def test_environment_transition_includes_environment_score_and_breakdown(self):
        environment = ParkingEnvironment(seed=3)

        _state, environment_score = environment.step({"action": "none"})
        transition = environment.get_last_transition()

        self.assertEqual(transition["environment_score"], environment_score)
        self.assertIn("step_breakdown", transition)
        self.assertEqual(len(transition["step_breakdown"]), 5)

    def test_environment_validates_invalid_redirect_zone(self):
        environment = ParkingEnvironment(seed=2)

        with self.assertRaises(ValueError):
            environment.step({"action": "redirect", "from": "Unknown", "to": "Library", "vehicles": 2})

    def test_policy_agent_handles_empty_state_and_logs_noop(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "policy_trace.json"),
        )
        agent = PolicyAgent(["A", "B"], logger=logger)

        decision = agent.decide({}, {}, {})

        self.assertEqual(decision["action"], "none")
        self.assertIn("empty or invalid state", decision["reason"])
        self.assertEqual(logger.get_by_event("policy_decision")[0]["level"], "INFO")

    def test_policy_agent_builds_rl_informed_decision_with_confidence(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "policy_rl_trace.json"),
        )
        agent = PolicyAgent(["A", "B", "C"], logger=logger)
        state = {
            "A": {"total_slots": 100, "occupied": 45, "free_slots": 55, "entry": 1, "exit": 3},
            "B": {"total_slots": 100, "occupied": 95, "free_slots": 5, "entry": 8, "exit": 1},
            "C": {"total_slots": 100, "occupied": 30, "free_slots": 70, "entry": 1, "exit": 4},
        }
        demand = {"A": 12, "B": 80, "C": 10}
        insight = {"uncertainty": {"entropy": 0.6}}
        agent.q_agent.epsilon = 0.0
        state_index = agent.q_agent.get_state(agent._build_observation(state, demand, insight, {"focus_zone": "B"}))
        agent.q_agent.q_table[state_index][agent.zones.index("C")] = 2.5

        decision = agent.decide(state, demand, insight, event_context={"focus_zone": "B", "recommended_zone": "C"})

        self.assertEqual(decision["action"], "redirect")
        self.assertEqual(decision["from"], "B")
        self.assertEqual(decision["to"], "C")
        self.assertIn("rl_state_index", decision)
        self.assertIn("exploration_rate", decision)
        self.assertGreater(decision["confidence"], 0.3)

    def test_policy_agent_updates_q_table_with_failure_penalty(self):
        logger = TraceLogger(
            max_traces=10,
            storage_path=os.path.join(self.temp_dir.name, "policy_learning_trace.json"),
        )
        agent = PolicyAgent(["A", "B"], logger=logger)
        old_state = {
            "A": {"total_slots": 100, "occupied": 94, "free_slots": 6, "entry": 7, "exit": 1},
            "B": {"total_slots": 100, "occupied": 25, "free_slots": 75, "entry": 1, "exit": 3},
        }
        new_state = {
            "A": {"total_slots": 100, "occupied": 95, "free_slots": 5, "entry": 8, "exit": 1},
            "B": {"total_slots": 100, "occupied": 24, "free_slots": 76, "entry": 1, "exit": 4},
        }
        action = {"action": "redirect", "from": "A", "to": "B", "vehicles": 4}
        old_index = agent.q_agent.get_state(agent._build_observation(old_state, {"A": 60, "B": 10}, {"uncertainty": {"entropy": 0.8}}, {}))
        action_index = agent.zones.index("B")
        before = agent.q_agent.q_table[old_index][action_index]

        agent.update(
            old_state,
            action,
            reward=1.0,
            new_state=new_state,
            demand={"A": 60, "B": 10},
            insight={"uncertainty": {"entropy": 0.8}},
            execution_feedback={"success": False, "blocked_action": action},
        )

        after = agent.q_agent.q_table[old_index][action_index]
        self.assertNotEqual(before, after)
        self.assertEqual(logger.get_by_event("policy_learning_update")[0]["level"], "INFO")

    def test_policy_agent_tolerates_zone_mismatch(self):
        agent = PolicyAgent(["A", "B", "C"])
        state = {
            "A": {"total_slots": 100, "occupied": 92, "free_slots": 8, "entry": 6, "exit": 1},
            "B": {"total_slots": 100, "occupied": 35, "free_slots": 65, "entry": 2, "exit": 3},
        }
        demand = {"A": 50, "B": 10}

        decision = agent.decide(state, demand, {"uncertainty": {"entropy": 0.4}})

        self.assertIn(decision["action"], {"redirect", "none"})
        if decision["action"] == "redirect":
            self.assertIn(decision["to"], state)

    def test_reward_agent_rewards_demand_aware_improvement(self):
        agent = RewardAgent()
        old_state = {
            "A": {"total_slots": 100, "occupied": 96, "free_slots": 4, "entry": 8, "exit": 1},
            "B": {"total_slots": 100, "occupied": 45, "free_slots": 55, "entry": 2, "exit": 3},
        }
        new_state = {
            "A": {"total_slots": 100, "occupied": 88, "free_slots": 12, "entry": 4, "exit": 6},
            "B": {"total_slots": 100, "occupied": 48, "free_slots": 52, "entry": 3, "exit": 2},
        }

        reward = agent.evaluate(
            old_state,
            new_state,
            action={"action": "redirect", "from": "A", "to": "B", "vehicles": 4},
            demand={"A": 80, "B": 15},
            event_context={"focus_zone": "A", "recommended_zone": "B"},
            kpis={"queue_length": 2, "estimated_search_time_min": 3.0, "allocation_success_pct": 100.0, "congestion_hotspots": 0},
            transition={"transfer_detail": {"moved": 4, "requested": 4}},
        )

        self.assertGreater(reward, 0)

    def test_reward_agent_penalizes_unnecessary_failed_redirect(self):
        agent = RewardAgent()
        state = {
            "A": {"total_slots": 100, "occupied": 60, "free_slots": 40, "entry": 1, "exit": 2},
            "B": {"total_slots": 100, "occupied": 58, "free_slots": 42, "entry": 1, "exit": 1},
        }

        reward = agent.evaluate(
            state,
            state,
            action={"action": "redirect", "from": "A", "to": "B", "vehicles": 3},
            demand={"A": 8, "B": 7},
            event_context={},
            kpis={"queue_length": 0, "estimated_search_time_min": 2.1, "allocation_success_pct": 0.0, "congestion_hotspots": 0},
            transition={"transfer_detail": {"moved": 0, "requested": 3}},
        )

        self.assertLess(reward, 0)

    def test_planner_prefers_live_hotspot_over_event_focus_for_redirect_source(self):
        planner = PlannerAgent()
        state = {
            "A": {"total_slots": 100, "occupied": 40, "free_slots": 60, "entry": 1, "exit": 2},
            "B": {"total_slots": 100, "occupied": 95, "free_slots": 5, "entry": 6, "exit": 1},
            "C": {"total_slots": 100, "occupied": 30, "free_slots": 70, "entry": 1, "exit": 3},
        }
        demand = {"A": 2, "B": 18, "C": 4}
        insight = {"confidence": 0.86}
        tools = {
            "get_goal_status": lambda: {},
            "build_zone_pressure_report": lambda current_state, current_demand: {
                zone: {"free_slots": current_state[zone]["free_slots"], "demand_pressure": current_demand.get(zone, 0)}
                for zone in current_state
            },
            "get_recent_cycles": lambda: [],
            "get_event_context": lambda: {
                "name": "Sports Event",
                "severity": "high",
                "focus_zone": "A",
                "recommended_zone": "C",
                "allocation_strategy": "Event-priority routing",
            },
            "get_scenario_mode": lambda: "Sports Event",
            "suggest_best_zone": lambda current_state: "C",
            "get_learning_profile": lambda **_kwargs: {
                "global_transfer_bias": 1.0,
                "scenario_profile": {"preferred_transfer_bias": 1.0},
                "route_profile": {"success_bias": 1.0},
            },
            "estimate_transfer_capacity": lambda from_zone, to_zone, requested: requested if (from_zone, to_zone) == ("B", "C") else 0,
        }

        with patch(
            "agents.planner_agent.ask_llm_for_structured_json",
            side_effect=lambda _agent_name, _context, _schema_text, fallback, system_instruction=None: fallback,
        ):
            plan = planner.plan(state, demand, insight, {"steps": 0}, tools)

        self.assertEqual(plan["proposed_action"]["from"], "B")
        self.assertEqual(plan["proposed_action"]["to"], "C")
        self.assertEqual(plan["goal"]["priority_zone"], "B")

    def test_planner_generates_multi_step_alternatives_scoring_and_feedback(self):
        planner = PlannerAgent()
        state = {
            "A": {"total_slots": 100, "occupied": 90, "free_slots": 10, "entry": 7, "exit": 1},
            "B": {"total_slots": 100, "occupied": 96, "free_slots": 4, "entry": 8, "exit": 1},
            "C": {"total_slots": 100, "occupied": 45, "free_slots": 55, "entry": 2, "exit": 4},
            "D": {"total_slots": 100, "occupied": 50, "free_slots": 50, "entry": 2, "exit": 2},
        }
        demand = {"A": 30, "B": 80, "C": 20, "D": 18}
        insight = {
            "confidence": 0.82,
            "uncertainty": {"entropy": 1.1, "confidence_gap": 0.32},
        }
        tools = {
            "get_goal_status": lambda: {},
            "build_zone_pressure_report": lambda current_state, current_demand: {
                zone: {"free_slots": current_state[zone]["free_slots"], "demand_pressure": current_demand.get(zone, 0)}
                for zone in current_state
            },
            "get_recent_cycles": lambda: [
                {"kpis": {"queue_length": 4}},
                {"kpis": {"queue_length": 5}},
            ],
            "get_event_context": lambda: {
                "name": "Exam Rush",
                "severity": "high",
                "focus_zone": "A",
                "recommended_zone": "C",
                "allocation_strategy": "Demand smoothing",
            },
            "get_operational_signals": lambda: {"queue_length": 5, "blocked_zone": None},
            "get_scenario_mode": lambda: "Exam Rush",
            "suggest_best_zone": lambda current_state: "C",
            "get_learning_profile": lambda **_kwargs: {
                "global_transfer_bias": 1.1,
                "scenario_profile": {"preferred_transfer_bias": 1.15},
                "route_profile": {"success_bias": 1.05},
            },
            "estimate_transfer_capacity": lambda _from, to_zone, requested: min(requested, 10 if to_zone == "C" else 7),
        }

        with patch(
            "agents.planner_agent.ask_llm_for_structured_json",
            side_effect=lambda _agent_name, _context, _schema_text, fallback, system_instruction=None: fallback,
        ):
            plan = planner.plan(state, demand, insight, {"steps": 3}, tools)

        self.assertIn("alternative_actions", plan)
        self.assertGreaterEqual(len(plan["alternative_actions"]), 1)
        self.assertIn("action_sequence", plan)
        self.assertEqual(len(plan["action_sequence"]), 3)
        self.assertIn("scoring", plan)
        self.assertIn("benefit_score", plan["scoring"])
        self.assertIn("risk_probability", plan["scoring"])
        self.assertIn("planner_feedback", plan)
        self.assertIn("temporal_reasoning", plan)
        self.assertIn("uncertainty_assessment", plan)
        self.assertGreaterEqual(plan["goal"]["horizon_steps"], 3)

    def test_planner_handles_missing_tools_without_crashing(self):
        planner = PlannerAgent()
        state = {
            "A": {"total_slots": 100, "occupied": 94, "free_slots": 6, "entry": 7, "exit": 1},
            "B": {"total_slots": 100, "occupied": 35, "free_slots": 65, "entry": 1, "exit": 3},
        }
        demand = {"A": 60, "B": 10}
        insight = {"confidence": 0.8, "uncertainty": {"entropy": 0.9}}

        with patch(
            "agents.planner_agent.ask_llm_for_structured_json",
            side_effect=lambda _agent_name, _context, _schema_text, fallback, system_instruction=None: fallback,
        ):
            plan = planner.plan(state, demand, insight, {"steps": 0}, {})

        self.assertIn("proposed_action", plan)
        self.assertIn("tool_calls", plan)
        self.assertTrue(all("tool" in call and "used" in call for call in plan["tool_calls"]))
        self.assertIn(plan["proposed_action"]["action"], {"redirect", "none"})

    def test_runtime_answers_zone_by_zone_query(self):
        runtime = ParkingRuntimeService(
            storage_path=self.runtime_path,
            memory_storage_path=self.memory_path,
            notification_storage_path=self.notification_path,
        )
        runtime.step()
        answer = runtime.ask("tell all the slots occupied in each block")["answer"]

        self.assertIn("Occupied slots by zone", answer)
        self.assertIn("Academic Block", answer)
        self.assertIn("Library", answer)

    def test_runtime_exposes_mock_notifications(self):
        runtime = ParkingRuntimeService(
            storage_path=self.runtime_path,
            memory_storage_path=self.memory_path,
            notification_storage_path=self.notification_path,
        )
        result = runtime.step()
        dispatch = result.get("notification_dispatch", [])

        self.assertGreater(len(dispatch), 0)
        self.assertIn(dispatch[0]["channel"], {"mobile_app", "sms_gateway", "campus_signage"})

    def test_runtime_reset_with_clear_memory_clears_learning_state(self):
        runtime = ParkingRuntimeService(
            storage_path=self.runtime_path,
            memory_storage_path=self.memory_path,
            notification_storage_path=self.notification_path,
        )
        runtime.step()
        before_reset = runtime.memory.get_learning_profile()
        self.assertNotEqual(before_reset["global_transfer_bias"], 1.0)

        runtime.reset(clear_memory=True)
        after_reset = runtime.memory.get_learning_profile()
        self.assertEqual(after_reset["global_transfer_bias"], 1.0)
        self.assertEqual(runtime.memory.get_metrics()["steps"], 0)

    def test_api_includes_notification_endpoint(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/notifications", paths)
        self.assertIn("/state", paths)
        self.assertIn("/benchmark", paths)
        self.assertIn("/metrics", paths)
        self.assertIn("/learning", paths)
        self.assertIn("/decision", paths)
        self.assertIn("/agents", paths)
        self.assertIn("/autonomy/start", paths)
        self.assertIn("/autonomy/stop", paths)
        self.assertIn("/autonomy/status", paths)
        self.assertIn("/capabilities", paths)
        self.assertIn("/visualization", paths)

    def test_agent_manager_validates_empty_input(self):
        with self.assertRaises(AgentManagerError):
            preprocess_user_input("   ")

    def test_agent_manager_exposes_loop_goal_memory_and_learning(self):
        original_runtime = agent_manager.runtime_service
        agent_manager.runtime_service = ParkingRuntimeService(
            storage_path=self.runtime_path,
            memory_storage_path=self.memory_path,
            notification_storage_path=self.notification_path,
        )
        try:
            goal = set_goal(
                {
                    "objective": "Keep search time under four minutes.",
                    "target_congested_zones": 1,
                    "horizon_steps": 5,
                    "target_search_time_min": 4.0,
                }
            )
            self.assertEqual(goal["objective"], "Keep search time under four minutes.")

            loop_result = run_agent_loop(steps=1)
            self.assertEqual(loop_result["steps"], 1)
            self.assertIn("Observe -> Plan", loop_result["loop"])

            memory_report = get_memory_report()
            self.assertIn("short_term_memory", memory_report)
            self.assertIn("long_term_memory", memory_report)

            learning_report = get_policy_learning_report()
            self.assertIn("q_table_shape", learning_report)
            self.assertIn("recent_rewards", learning_report)
        finally:
            agent_manager.runtime_service = original_runtime

    def test_trace_logger_supports_levels_filters_limits_and_persistence(self):
        trace_path = os.path.join(self.temp_dir.name, "trace_log.json")
        logger = TraceLogger(max_traces=2, storage_path=trace_path)

        first = logger.log(1, "observe", {"zones": 5}, level="INFO")
        logger.warning(2, "decide", {"risk": "medium"})
        logger.error(3, "act", {"failed": True})

        self.assertIn("time", first)
        self.assertEqual(first["level"], "INFO")
        self.assertEqual(len(logger.get_traces()), 2)
        self.assertEqual(logger.get_traces(step=2)[0]["event"], "decide")
        self.assertEqual(logger.get_traces(level="ERROR")[0]["event"], "act")
        self.assertIn("counts_by_level", logger.summary())
        self.assertIn("ERROR", logger.pretty(limit=1)[0])

        reloaded = TraceLogger(max_traces=5, storage_path=trace_path)
        self.assertEqual(len(reloaded.get_traces()), 2)
        self.assertEqual(reloaded.get_by_event("act")[0]["level"], "ERROR")

    def test_runtime_benchmark_produces_agentic_vs_baseline_summary(self):
        runtime = ParkingRuntimeService(
            storage_path=self.runtime_path,
            memory_storage_path=self.memory_path,
            notification_storage_path=self.notification_path,
        )
        benchmark = runtime.run_benchmark(episodes=1, steps_per_episode=4)

        self.assertIn("aggregate", benchmark)
        self.assertIn("scenarios", benchmark)
        self.assertGreater(len(benchmark["scenarios"]), 0)
        first = benchmark["scenarios"][0]
        self.assertIn("agentic", first)
        self.assertIn("baseline", first)
        self.assertIn("delta_search_time", first)

    def test_message_bus_structures_messages_and_supports_unsubscribe(self):
        bus = MessageBus(max_messages=3)

        class Receiver:
            def __init__(self):
                self.received = []

            def receive(self, topic, message):
                self.received.append((topic, message))

        receiver = Receiver()
        bus.subscribe("planning", receiver)
        result = bus.publish("planning", {"goal": "rebalance"}, sender="PlannerAgent", message_type="plan", priority="high")

        self.assertTrue(result["published"])
        self.assertEqual(len(receiver.received), 1)
        delivered = receiver.received[0][1]
        self.assertEqual(delivered["sender"], "PlannerAgent")
        self.assertEqual(delivered["type"], "plan")
        self.assertEqual(delivered["priority"], "high")
        self.assertIn("timestamp", delivered)
        self.assertTrue(bus.unsubscribe("planning", receiver))

    def test_message_bus_isolates_delivery_errors_and_bounds_history(self):
        bus = MessageBus(max_messages=2)

        class GoodReceiver:
            def __init__(self):
                self.count = 0

            def receive(self, topic, message):
                self.count += 1

        class BadReceiver:
            def receive(self, topic, message):
                raise RuntimeError("boom")

        good = GoodReceiver()
        bad = BadReceiver()
        bus.subscribe("system", good)
        bus.subscribe("system", bad)
        bus.publish("system", "one")
        bus.publish("system", "two")
        bus.publish("system", "three")

        self.assertEqual(good.count, 3)
        self.assertEqual(len(bus.get_messages()), 2)
        self.assertEqual(bus.get_delivery_errors(limit=1)[0]["error"], "boom")

    def test_message_bus_validates_agents_and_supports_selective_delivery(self):
        bus = MessageBus()

        class Receiver:
            def __init__(self, name):
                self.name = name
                self.received = []

            def receive(self, topic, message):
                self.received.append(message)

        alpha = Receiver("alpha")
        beta = Receiver("beta")
        bus.subscribe("reward", alpha)
        bus.subscribe("reward", beta)

        with self.assertRaises(TypeError):
            bus.subscribe("reward", object())

        bus.publish("reward", {"score": 1.2}, target_agents=["beta"])
        self.assertEqual(len(alpha.received), 0)
        self.assertEqual(len(beta.received), 1)

    def test_message_bus_publish_async_supports_async_receivers(self):
        bus = MessageBus()

        class AsyncReceiver:
            def __init__(self):
                self.received = []

            async def receive(self, topic, message):
                self.received.append((topic, message["sender"]))

        receiver = AsyncReceiver()
        bus.subscribe("policy", receiver)

        result = asyncio.run(bus.publish_async("policy", {"action": "hold"}, sender="PolicyAgent"))

        self.assertTrue(result["published"])
        self.assertEqual(receiver.received[0], ("policy", "PolicyAgent"))

    def test_simulation_logger_batches_rows_and_writes_structured_csv(self):
        log_dir = os.path.join(self.temp_dir.name, "simulation_logs")
        logger = SimulationLogger(log_dir=log_dir, batch_size=2, max_in_memory=10, max_file_rows=20)

        first = logger.log_step({"step_number": 1, "mode": "agentic_loop", "action": {"action": "none"}})
        self.assertIn("timestamp", first)
        self.assertFalse(os.path.exists(logger.log_file))

        logger.log_step({"step_number": 2, "mode": "goal_hold", "action": {"action": "redirect", "to": "B"}})
        self.assertTrue(os.path.exists(logger.log_file))

        with open(logger.log_file, "r", encoding="utf-8") as file:
            contents = file.read()
        self.assertIn("timestamp", contents)
        self.assertIn("log_type", contents)
        self.assertIn("goal_hold", contents)

    def test_simulation_logger_flush_status_and_row_limit(self):
        log_dir = os.path.join(self.temp_dir.name, "simulation_logs_limited")
        logger = SimulationLogger(log_dir=log_dir, batch_size=10, max_in_memory=5, max_file_rows=3)

        for step in range(5):
            logger.log_event({"step_number": step + 1, "status": "ok", "error": ""}, log_type="event")

        self.assertEqual(logger.get_status()["buffered_records"], 5)
        self.assertTrue(logger.flush())

        with open(logger.log_file, "r", encoding="utf-8") as file:
            rows = file.readlines()
        self.assertEqual(len(rows), 4)
        self.assertLessEqual(len(logger.get_logs()), 5)

    def test_predict_demand_details_validates_inputs(self):
        with self.assertRaises(ValueError):
            demand_predictor.predict_demand_details(25, 2, 0, 0)
        with self.assertRaises(ValueError):
            demand_predictor.predict_demand_details(10, 9, 0, 0)

    def test_predict_demand_details_reports_fallback_metadata(self):
        with patch("ml.predict._load_model", return_value=None):
            details = demand_predictor.predict_demand_details(9, 2, 1, 0)

        self.assertIn("prediction", details)
        self.assertIn("confidence", details)
        self.assertTrue(details["fallback_used"])
        self.assertEqual(details["mode"], "fallback")

    def test_predict_demand_batch_supports_multiple_records(self):
        with patch("ml.predict._load_model", return_value=None):
            results = demand_predictor.predict_demand_batch(
                [
                    {"hour": 8, "day": 1, "zone_id": 0, "vehicle_type": 0},
                    {"hour": 18, "day": 6, "zone_id": 2, "vehicle_type": 1},
                ]
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all("prediction" in item and "mode" in item for item in results))

    def test_train_model_loader_creates_day_of_week_features(self):
        df = train_model_module._load_dataset()

        self.assertIn("day_of_week", df.columns)
        self.assertIn("is_weekend", df.columns)
        self.assertIn("net_flow", df.columns)
        self.assertIn("occupancy_ratio", df.columns)

    def test_train_model_returns_metrics_and_versioned_outputs(self):
        model_dir = os.path.join(self.temp_dir.name, "models")
        model_path = os.path.join(model_dir, "demand_model.pkl")
        metrics_path = os.path.join(model_dir, "demand_model_metrics.json")
        versioned_model_path = os.path.join(model_dir, "versioned.pkl")

        with patch.object(train_model_module, "model_dir", model_dir), \
            patch.object(train_model_module, "model_path", model_path), \
            patch.object(train_model_module, "metrics_path", metrics_path), \
            patch.object(train_model_module, "versioned_model_path", versioned_model_path):
            metadata = train_model_module.train_model()

        self.assertIn("metrics", metadata)
        self.assertIn("mae", metadata["metrics"])
        self.assertIn("rmse", metadata["metrics"])
        self.assertIn("r2", metadata["metrics"])
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(metrics_path))


if __name__ == "__main__":
    unittest.main()
