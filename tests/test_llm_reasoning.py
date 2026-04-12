import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm_reasoning
from tools import get_tools
from environment.parking_environment import ParkingEnvironment


class LLMReasoningTests(unittest.TestCase):
    def test_env_path_resolves_from_project_root(self):
        self.assertEqual(llm_reasoning.ENV_PATH, Path(llm_reasoning.__file__).resolve().parent / ".env")
        self.assertEqual(llm_reasoning.ENV_EXAMPLE_PATH, Path(llm_reasoning.__file__).resolve().parent / ".env.example")

    def test_connectivity_errors_are_classified(self):
        self.assertTrue(llm_reasoning._is_connectivity_llm_error("ConnectError: [Errno 8] nodename nor servname provided"))
        self.assertTrue(llm_reasoning._is_connectivity_llm_error("SSL certificate verify failed"))
        self.assertFalse(llm_reasoning._is_connectivity_llm_error("PERMISSION_DENIED"))

    def test_create_llm_agent_falls_back_when_live_llm_unavailable(self):
        tools = get_tools(ParkingEnvironment())
        agent = llm_reasoning.create_llm_agent(tools)

        with patch("llm_reasoning.get_llm", return_value=None):
            result = agent.invoke("Find the best parking zone")

        self.assertIn("Best zone:", result["output"])
        self.assertIn("State Snapshot:", result["output"])

    def test_create_llm_agent_uses_live_llm_when_available(self):
        tools = get_tools(ParkingEnvironment())
        agent = llm_reasoning.create_llm_agent(tools)
        fake_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="Use Zone B for arrivals."))

        with patch("llm_reasoning.get_llm", return_value=fake_llm):
            result = agent.invoke("Find the best parking zone")

        self.assertEqual(result["output"], "Use Zone B for arrivals.")

    def test_operational_reasoning_is_labeled_as_local_summary(self):
        state = {
            "A": {"total_slots": 100, "occupied": 90, "free_slots": 10, "entry": 4, "exit": 1},
            "B": {"total_slots": 100, "occupied": 40, "free_slots": 60, "entry": 1, "exit": 3},
        }

        reasoning = llm_reasoning.get_operational_reasoning(state)

        self.assertEqual(reasoning["source"], "local_operational_summary")
        self.assertIn("redirect incoming arrivals", reasoning["text"])


if __name__ == "__main__":
    unittest.main()
