import os
import unittest
import tempfile
import asyncio
from logs.logger import SimulationLogger
from adk.trace_logger import TraceLogger
from communication.message_bus import MessageBus
from ui.logic.state_manager import CacheManager

class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    # --- Message Bus Tests ---
    
    def test_message_bus_pub_sub_async_concurrency(self):
        bus = MessageBus()
        class AsyncReceiver:
            def __init__(self):
                self.received = []
            async def receive(self, topic, message):
                self.received.append((topic, message["sender"]))

        receiver = AsyncReceiver()
        bus.subscribe("system", receiver)
        
        async def burst():
            return await asyncio.gather(
                bus.publish_async("system", "ping1", sender="AgentA"),
                bus.publish_async("system", "ping2", sender="AgentB"),
                bus.publish_async("system", "ping3", sender="AgentC")
            )
            
        asyncio.run(burst())
        self.assertEqual(len(receiver.received), 3)

    def test_message_bus_delivery_error_isolation(self):
        bus = MessageBus()
        class BadReceiver:
            def receive(self, topic, message):
                raise ValueError("Intentional Failure")
                
        receiver = BadReceiver()
        bus.subscribe("system", receiver)
        
        # Ensure it doesn't crash the publisher thread
        result = bus.publish("system", "payload")
        self.assertTrue(result["published"])
        self.assertEqual(len(bus.get_delivery_errors()), 1)
        self.assertIn("Intentional Failure", bus.get_delivery_errors()[0]["error"])

    def test_message_bus_rejects_invalid_subscriber_structure(self):
        bus = MessageBus()
        with self.assertRaises(TypeError):
            bus.subscribe("system", lambda x: print(x)) # Doesn't have .receive()

    # --- Trace Logger Tests ---

    def test_trace_logger_persistence_and_filtering(self):
        path = os.path.join(self.temp_dir.name, "trace.json")
        logger = TraceLogger(storage_path=path)
        logger.log(1, "test_event", {"data": 1}, level="WARN")
        logger.log(2, "error_event", {"data": 2}, level="ERROR")
        
        # Verify persistence immediately works
        tracer2 = TraceLogger(storage_path=path)
        self.assertEqual(len(tracer2.get_traces()), 2)
        
        # Filter check
        self.assertEqual(len(tracer2.get_traces(level="WARN")), 1)

    # --- Simulation Logger Tests ---

    def test_simulation_logger_csv_batching_limit(self):
        path = os.path.join(self.temp_dir.name, "sim_logger")
        logger = SimulationLogger(log_dir=path, batch_size=2)
        
        logger.log_step({"step_number": 1, "mode": "M", "action": {}})
        # shouldn't write to disk yet because batch is 2
        file_path = logger.log_file
        self.assertFalse(os.path.exists(file_path))
        
        logger.log_step({"step_number": 2, "mode": "M", "action": {}})
        # Should flush
        self.assertTrue(os.path.exists(file_path))

    def test_state_frame_uses_backend_capacity_as_single_source(self):
        frame = CacheManager.get_state_frame({
            "Main Block": {
                "capacity": 420,
                "total_slots": 0,
                "car_slots": 120,
                "bike_slots": 300,
                "occupied": 110,
                "free_slots": 310,
                "entry": 1,
                "exit": 2,
            },
            "Legacy Bad Block": {
                "capacity": 0,
                "car_slots": 0,
                "bike_slots": 300,
                "occupied": 110,
                "free_slots": 0,
                "entry": 0,
                "exit": 0,
            },
        }, step=1)

        main = frame[frame["Zone"] == "Main Block"].iloc[0]
        self.assertEqual(list(frame.columns).count("Capacity"), 1)
        self.assertEqual(int(main["Capacity"]), 420)
        self.assertEqual(int(main["Car Slots"]) + int(main["Bike Slots"]), 420)
        self.assertEqual(int(main["Occupied"]) + int(main["Free"]), 420)

        repaired = frame[frame["Zone"] == "Legacy Bad Block"].iloc[0]
        self.assertEqual(int(repaired["Capacity"]), 300)
        self.assertEqual(int(repaired["Occupied"]) + int(repaired["Free"]), 300)
        
if __name__ == "__main__":
    unittest.main()
