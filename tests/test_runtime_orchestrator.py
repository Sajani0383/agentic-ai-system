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
        self.assertIn("budget_level", result["reasoning_budget"])
        self.assertTrue(any(item.get("agent") == "ReasoningBudget" for item in result.get("agent_interactions", [])))

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
        self.assertIn("notification_summary", snapshot)
        self.assertIn("benchmark_summary", snapshot)
        self.assertIn("last_llm_decision", snapshot)
        self.assertIn("llm_usage_summary", snapshot)

        runtime.step()
        stepped = runtime.get_runtime_snapshot()
        self.assertTrue(len(stepped["agent_loop_steps"]) >= 5)
        self.assertIn("decision", stepped["reasoning_summary"])
        self.assertIn("history", stepped["memory_summary"])
        self.assertIn("local_reasoning", stepped["llm_usage_summary"])
        self.assertIn("gemini_attempts", stepped["llm_usage_summary"])

    def test_reset_clears_llm_backoff_state(self):
        STATUS_MANAGER.start_backoff("ConnectError: test", seconds=30, kind="transient")
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.reset(clear_memory=False)
        snapshot = runtime.get_runtime_snapshot()
        self.assertFalse(snapshot["llm_status"]["quota_backoff"]["active"])

if __name__ == "__main__":
    unittest.main()
