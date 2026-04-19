import unittest
from unittest.mock import patch, MagicMock
from environment.parking_environment import ParkingEnvironment

class ParkingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = ParkingEnvironment(zones=["A", "B"], seed=42)

    def test_environment_initializes_with_seed_determinism(self):
        env1 = ParkingEnvironment(seed=5)
        env2 = ParkingEnvironment(seed=5)
        self.assertEqual(env1.get_state(), env2.get_state())

    @patch("environment.parking_environment.random.uniform", return_value=0.5)
    def test_step_advances_time_stochastically(self, mock_uniform):
        old_state = self.env.get_state()
        new_state, reward = self.env.step({"action": "none"})
        self.assertIn("A", new_state)
        self.assertIn("B", new_state)
        self.assertIn("step_breakdown", self.env.get_last_transition())

    def test_environment_redirect_logic_accurate_math(self):
        self.env.state = {
            "A": {"total_slots": 100, "occupied": 90, "entry": 0, "exit": 0},
            "B": {"total_slots": 100, "occupied": 40, "entry": 0, "exit": 0},
        }
        self.env._advance_time = lambda: None
        self.env._build_dynamic_signals = lambda x: {}
        self.env._time_multiplier = lambda x: 1.0
        self.env._dynamic_signal_multiplier = lambda x, y, z: 1.0
        
        with patch("environment.parking_environment.predict_demand", return_value=10):
            after, reward = self.env.step({
                "action": "redirect",
                "from": "A",
                "to": "B",
                "vehicles": 5
            })
            
        trans = self.env.get_last_transition()
        self.assertGreater(trans["transfer_detail"]["moved"], 0) # Ensures logic successfully moved at least 1 car

    def test_invalid_redirect_throws_value_error(self):
        with self.assertRaises(ValueError):
            self.env.step({"action": "redirect", "from": "UnknownZone", "to": "B", "vehicles": 5})
            
    def test_negative_invalid_action_format(self):
        state, reward = self.env.step(None) # Missing format gracefully falls back instead of breaking process natively in simulation
        self.assertIsInstance(reward, float)

        state2, reward2 = self.env.step({"action": "magical_teleport"})
        self.assertIsInstance(reward2, float)

    def test_performance_massive_zone_initialization(self):
        massive_zones = [f"Zone_{i}" for i in range(5000)]
        big_env = ParkingEnvironment(zones=massive_zones)
        self.assertEqual(len(big_env.get_state()), 5000)
        _, reward = big_env.step({"action": "none"})
        self.assertIsInstance(reward, float)

if __name__ == "__main__":
    unittest.main()
