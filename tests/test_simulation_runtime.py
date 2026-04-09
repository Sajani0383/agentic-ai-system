import os
import tempfile
import unittest

from adk.agent_api import app
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
        self.assertTrue(memory.get_q_table() is not None)

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


if __name__ == "__main__":
    unittest.main()
