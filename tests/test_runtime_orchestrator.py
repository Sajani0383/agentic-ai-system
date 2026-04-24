import os
import unittest
import tempfile
import asyncio
from unittest.mock import patch

from services.parking_runtime import ParkingRuntimeService
from agent_controller import AgentController
from environment.parking_environment import ParkingEnvironment
from agent_memory import AgentMemory
from llm.client import STATUS_MANAGER

class RuntimeOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mem_path = os.path.join(self.temp_dir.name, "mem.json")
        self.state_path = os.path.join(self.temp_dir.name, "state.json")
        self.notif_path = os.path.join(self.temp_dir.name, "notif.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Agent Controller ---
    def test_agent_controller_maintains_schema(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        
        result = controller.step()
        self.assertIn("step_number", result)
        self.assertIn("execution_output", result)
        self.assertIn("planner_output", result)
        self.assertIn("critic_output", result)

    # --- Parking Runtime Concurrency & Validation ---
    def test_runtime_handles_async_concurrency_cleanly(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        
        async def concurrent_steps():
            return await asyncio.gather(
                runtime.async_step(),
                runtime.async_step(),
                runtime.async_step(),
                runtime.async_step(),
                runtime.async_step()
            )
        
        results = asyncio.run(concurrent_steps())
        self.assertEqual(len(results), 5)
        steps_executed = [r["step_number"] for r in results]
        self.assertEqual(len(set(steps_executed)), 5)

    def test_runtime_rejects_corrupted_inputs_cleanly(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        
        bad_scenario_types = [None, 123, "", [], {}]
        for bad_input in bad_scenario_types:
            snapshot = runtime.set_scenario_mode(bad_input)
            self.assertIn("scenario_mode", snapshot)
            # The current default default value if unset or reset is likely Auto Schedule based on runtime initialization
            self.assertEqual(snapshot["scenario_mode"], "Auto Schedule") 

    def test_runtime_ask_recovers_empty_queries(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.step()
        
        resp1 = runtime.ask("")
        self.assertEqual(resp1["answer"], "Invalid or empty query.")
        
        resp2 = runtime.ask("free slots in each zone")
        self.assertIn("Free slots by zone", resp2["answer"])

    def test_agent_controller_exposes_reasoning_budget_and_interactions(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)

        result = controller.step()
        self.assertIn("reasoning_budget", result)
        self.assertIn("reasoning_summary", result)
        self.assertIn("agent_loop_steps", result)
        self.assertIn("memory_summary", result)
        self.assertIn("decision_provenance", result)
        self.assertIn("budget_level", result["reasoning_budget"])
        self.assertTrue(any(item.get("agent") == "ReasoningBudget" for item in result.get("agent_interactions", [])))

    def test_controller_autonomously_revises_goal_metadata(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)

        result = controller.step()
        goal = result.get("goal", {})
        self.assertTrue(goal)
        self.assertEqual(goal.get("source"), "autonomous_controller")
        self.assertIn("revision_reason", goal)
        self.assertGreaterEqual(int(goal.get("revision_count", 0) or 0), 1)

    @patch("services.parking_runtime.get_operational_briefing")
    def test_runtime_passes_briefing_budget_flag(self, mock_briefing):
        mock_briefing.return_value = {"headline": "Local", "narrative": "", "prediction": "", "suggestions": [], "decision_commentary": ""}
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.step()
        self.assertTrue(mock_briefing.called)
        self.assertIn("use_llm", mock_briefing.call_args.kwargs)

    def test_runtime_snapshot_exposes_tab_ready_structures(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        snapshot = runtime.get_runtime_snapshot()
        self.assertIn("reasoning_summary", snapshot)
        self.assertIn("agent_loop_steps", snapshot)
        self.assertIn("memory_summary", snapshot)
        self.assertIn("decision_provenance", snapshot.get("latest_result", {}))
        self.assertIn("notification_summary", snapshot)
        self.assertIn("benchmark_summary", snapshot)
        self.assertIn("last_llm_decision", snapshot)
        self.assertIn("llm_usage_summary", snapshot)

        runtime.step()
        stepped = runtime.get_runtime_snapshot()
        self.assertTrue(len(stepped["agent_loop_steps"]) >= 5)
        self.assertIn("decision", stepped["reasoning_summary"])
        self.assertIn("final_authority", stepped["reasoning_summary"])
        self.assertIn("history", stepped["memory_summary"])
        self.assertIn("goal_history", stepped["memory_summary"])
        self.assertIn("local_reasoning", stepped["llm_usage_summary"])
        self.assertIn("gemini_attempts", stepped["llm_usage_summary"])

    def test_memory_can_persist_llm_route_rules(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.set_scenario_mode("Exam Rush")
        runtime.memory.record_llm_rule(
            "Exam Rush",
            {
                "llm_advisory_used": True,
                "llm_influence": True,
                "llm_source": "gemini",
                "llm_summary": "Prefer Tech Park when Library pressure rises.",
            },
            {"action": "redirect", "from": "Library", "to": "Tech Park", "vehicles": 2},
            0.45,
            {},
        )
        snapshot = runtime.get_runtime_snapshot()
        rules = snapshot["memory_summary"]["learning_profile"].get("llm_memory_rules", [])
        self.assertTrue(rules)
        self.assertEqual(rules[0]["route_key"], "Library->Tech Park")

    def test_micro_redirect_prefers_llm_memory_destination(self):
        env = ParkingEnvironment(zones=["A", "B", "C"])
        env.state = {
            "A": {"total_slots": 100, "occupied": 96, "free_slots": 4, "entry": 6, "exit": 1},
            "B": {"total_slots": 100, "occupied": 35, "free_slots": 65, "entry": 2, "exit": 3},
            "C": {"total_slots": 100, "occupied": 20, "free_slots": 80, "entry": 1, "exit": 2},
        }
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        context = {
            "state": env.state,
            "demand": {"A": 20, "B": 3, "C": 4},
            "learning_profile": {
                "blocked_routes": [],
                "llm_memory_rules": [
                    {"scenario": "Exam Rush", "from": "A", "to": "B", "route_key": "A->B", "strength": 0.95, "ttl": 8}
                ],
            },
        }
        action = controller._build_micro_redirect(context, "test")
        self.assertEqual(action["to"], "B")
        self.assertGreaterEqual(action["vehicles"], 2)

    def test_sequence_continuation_uses_follow_up_steps(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        mem.log_cycle({"step": 1, "kpis": {"queue_length": 1}, "reward": {"agentic_reward_score": 0.1}})
        mem.set_goal(
            {
                "objective": "Test sequence continuation",
                "pending_action_sequence": [
                    {"step": 1, "phase": "stabilize", "action": {"action": "redirect", "from": "A", "to": "B", "vehicles": 2}},
                    {"step": 2, "phase": "monitor", "action": {"action": "observe"}},
                    {"step": 3, "phase": "fallback", "action": {"action": "redirect", "from": "A", "to": "B", "vehicles": 1}},
                ],
                "pending_sequence_index": 1,
            }
        )
        continued = controller._apply_sequence_continuation(
            {"proposed_action": {"action": "redirect", "from": "A", "to": "B", "vehicles": 2}},
            {"goal": mem.get_active_goal(), "operational_signals": {"queue_length": 1}},
        )
        self.assertTrue(continued.get("sequence_continuation_applied"))
        self.assertEqual(continued["proposed_action"]["action"], "none")
        self.assertEqual(continued.get("sequence_step_executed"), 2)

    def test_reset_clears_llm_backoff_state(self):
        STATUS_MANAGER.start_backoff("ConnectError: test", seconds=30, kind="transient")
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.reset(clear_memory=False)
        snapshot = runtime.get_runtime_snapshot()
        self.assertFalse(snapshot["llm_status"]["quota_backoff"]["active"])

if __name__ == "__main__":
    unittest.main()
