import os
import tempfile
import unittest
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
from agent_controller import AgentController
from agent_memory import AgentMemory
from environment.parking_environment import ParkingEnvironment
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


if __name__ == "__main__":
    unittest.main()
