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
        self.assertIn("current_state", snapshot)
        self.assertIn("blocks", snapshot["current_state"])
        self.assertIn("vehicles", snapshot["current_state"])
        self.assertIn("actions", snapshot["current_state"])
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

    def test_shared_current_state_vehicle_counts_match_blocks(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.step()
        snapshot = runtime.get_runtime_snapshot()
        current_state = snapshot["current_state"]
        vehicles_by_block = {}
        for vehicle in current_state["vehicles"]:
            vehicles_by_block[vehicle["block"]] = vehicles_by_block.get(vehicle["block"], 0) + 1
            self.assertIn("position", vehicle)
            self.assertIn("x", vehicle["position"])
            self.assertIn("y", vehicle["position"])
        for block_name, block_data in current_state["blocks"].items():
            self.assertEqual(block_data["occupied"], vehicles_by_block.get(block_name, 0))

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

    def test_critical_critic_rejection_cannot_execute_redirect(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        planner_output = {
            "proposed_action": {
                "action": "redirect",
                "from": "A",
                "to": "B",
                "vehicles": 4,
                "force_micro": True,
            }
        }
        critic_output = {
            "approved": False,
            "risk_level": "high",
            "risk_score": 91.5,
            "risk_factors": {},
            "critic_notes": ["Safety Override: Critical risk (91.5). Reverting to system baseline."],
            "revised_action": {"action": "none"},
        }

        normalized = controller._validate_critic_contract(critic_output, planner_output)
        execution = controller.executor_agent.execute(normalized, env)
        execution = controller._validate_execution_contract(execution, normalized)

        self.assertFalse(normalized["approved"])
        self.assertEqual(normalized["revised_action"]["action"], "none")
        self.assertEqual(execution["final_action"]["action"], "none")
        self.assertEqual(execution["executed_vehicles"], 0)

    def test_approved_micro_action_removes_stop_language(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        planner_output = {
            "proposed_action": {
                "action": "redirect",
                "from": "A",
                "to": "B",
                "vehicles": 2,
                "force_micro": True,
            }
        }
        critic_output = {
            "approved": False,
            "risk_level": "high",
            "risk_score": 70,
            "risk_factors": {},
            "critic_notes": [
                "Safety Override: Critical risk (70). Reverting to system baseline.",
                "VETO: Expected gain is zero or negative.",
            ],
            "revised_action": {"action": "redirect", "from": "A", "to": "B", "vehicles": 2},
        }

        normalized = controller._validate_critic_contract(critic_output, planner_output)
        note_text = " ".join(normalized["critic_notes"])

        self.assertTrue(normalized["approved"])
        self.assertNotIn("Safety Override", note_text)
        self.assertNotIn("Reverting to system baseline", note_text)
        self.assertIn("Mitigated concern", note_text)

    def test_negative_reward_blocks_failed_route(self):
        mem = AgentMemory(storage_path=self.mem_path)
        mem.add_failure("Library", "Main Block", "Negative reward (-0.63): action worsened system.")
        profile = mem.get_learning_profile(from_zone="Library", to_zone="Main Block")

        self.assertIn("Library->Main Block", profile["blocked_routes"])
        self.assertGreater(profile["blocked_route_ttl"].get("Library->Main Block", 0), 0)

    def test_llm_avoid_rule_blocks_planner_route(self):
        env = ParkingEnvironment(zones=["Admin Block", "Basic Eng Lab", "Main Block"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        context = {
            "state": env.get_state(),
            "learning_profile": {
                "blocked_routes": [],
                "llm_memory_rules": [
                    {
                        "route_key": "Admin Block->Basic Eng Lab",
                        "from": "Admin Block",
                        "to": "Basic Eng Lab",
                        "strength": -0.5,
                        "avoid_count": 1,
                        "prefer_count": 0,
                    }
                ],
            },
        }
        plan = controller._validate_planner_contract(
            {
                "proposed_action": {
                    "action": "redirect",
                    "from": "Admin Block",
                    "to": "Basic Eng Lab",
                    "vehicles": 2,
                }
            },
            context,
        )

        self.assertEqual(plan["proposed_action"]["action"], "none")

    def test_micro_redirect_skips_llm_avoid_destination(self):
        env = ParkingEnvironment(zones=["Admin Block", "Basic Eng Lab", "Main Block"])
        env.state = {
            "Admin Block": {"total_slots": 100, "occupied": 96, "free_slots": 4, "entry": 8, "exit": 1},
            "Basic Eng Lab": {"total_slots": 100, "occupied": 10, "free_slots": 90, "entry": 1, "exit": 1},
            "Main Block": {"total_slots": 100, "occupied": 20, "free_slots": 80, "entry": 1, "exit": 1},
        }
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        action = controller._build_micro_redirect(
            {
                "state": env.state,
                "demand": {"Admin Block": 20, "Basic Eng Lab": 1, "Main Block": 1},
                "learning_profile": {
                    "blocked_routes": [],
                    "llm_memory_rules": [
                        {
                            "route_key": "Admin Block->Basic Eng Lab",
                            "from": "Admin Block",
                            "to": "Basic Eng Lab",
                            "strength": -0.75,
                            "avoid_count": 2,
                            "prefer_count": 0,
                        }
                    ],
                },
            },
            "test",
        )

        self.assertEqual(action["from"], "Admin Block")
        self.assertNotEqual(action["to"], "Basic Eng Lab")

    def test_critic_suggested_replan_is_used_when_required(self):
        env = ParkingEnvironment(zones=["Admin Block", "Basic Eng Lab", "Main Block"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        critic_output = {
            "approved": True,
            "revised_action": {"action": "redirect", "from": "Admin Block", "to": "Basic Eng Lab", "vehicles": 4},
            "replan_recommendation": {
                "required": True,
                "suggested_action": {"action": "redirect", "from": "Admin Block", "to": "Main Block", "vehicles": 2},
            },
        }

        self.assertTrue(controller._needs_replan(critic_output))
        action = controller._select_replan_action(
            critic_output,
            {"state": env.get_state(), "learning_profile": {"blocked_routes": []}},
        )

        self.assertEqual(action["to"], "Main Block")
        self.assertLessEqual(action["vehicles"], 2)

    def test_learning_gate_replaces_blocked_final_route_before_execution(self):
        env = ParkingEnvironment(zones=["Admin Block", "Basic Eng Lab", "Main Block"])
        env.state = {
            "Admin Block": {"total_slots": 100, "occupied": 96, "free_slots": 4, "entry": 8, "exit": 1},
            "Basic Eng Lab": {"total_slots": 100, "occupied": 10, "free_slots": 90, "entry": 1, "exit": 1},
            "Main Block": {"total_slots": 100, "occupied": 20, "free_slots": 80, "entry": 1, "exit": 1},
        }
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        planner_output = {
            "proposed_action": {"action": "redirect", "from": "Admin Block", "to": "Basic Eng Lab", "vehicles": 2}
        }
        critic_output = {
            "approved": True,
            "revised_action": {"action": "redirect", "from": "Admin Block", "to": "Basic Eng Lab", "vehicles": 2},
            "critic_notes": [],
        }
        _, gated = controller._enforce_learning_before_execution(
            planner_output,
            critic_output,
            {
                "state": env.state,
                "demand": {"Admin Block": 20, "Basic Eng Lab": 1, "Main Block": 1},
                "learning_profile": {"blocked_routes": ["Admin Block->Basic Eng Lab"]},
            },
        )

        self.assertEqual(gated["revised_action"]["action"], "redirect")
        self.assertNotEqual(gated["revised_action"]["to"], "Basic Eng Lab")

    def test_negative_reward_reduces_next_transfer_before_blocking(self):
        env = ParkingEnvironment(zones=["A", "B"])
        mem = AgentMemory(storage_path=self.mem_path)
        controller = AgentController(environment=env, memory=mem)
        planner_output = {
            "proposed_action": {"action": "redirect", "from": "A", "to": "B", "vehicles": 6, "confidence": 0.8}
        }
        controller._apply_controller_pressure_guard(
            planner_output,
            {"learning_profile": {"last_reward": -0.15}},
        )

        self.assertLessEqual(planner_output["proposed_action"]["vehicles"], 2)
        self.assertTrue(planner_output["proposed_action"]["reward_reduced"])

    def test_user_entry_updates_unified_vehicle_state(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        response = runtime.register_user_entry({
            "name": "Demo User",
            "vehicle_number": "TN01AB1234",
            "user_type": "student",
            "vehicle_type": "car",
            "preferred_block": "Main Block",
            "gate": "Gate1",
        })
        snapshot = runtime.get_runtime_snapshot()

        self.assertIn(response["status"], {"assigned", "redirected"})
        self.assertIn("user_vehicles", snapshot)
        self.assertTrue(any(vehicle.get("number") == "TN01AB1234" for vehicle in snapshot["user_vehicles"]))
        self.assertGreaterEqual(snapshot["vehicle_stats"]["user"], 1)
        self.assertTrue(any(event.get("event") == "entry" for event in snapshot["events"]))

    def test_user_exit_tracks_exit_event_with_dashboard_name(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.register_user_entry({
            "name": "Private User",
            "vehicle_number": "TN02CD5678",
            "user_type": "staff",
            "vehicle_type": "bike",
            "gate": "Gate2",
        })
        response = runtime.register_user_exit({"vehicle_number": "TN02CD5678"})
        snapshot = runtime.get_runtime_snapshot()

        self.assertEqual(response["notification"], "Exit completed")
        self.assertTrue(any(vehicle.get("status") == "exited" for vehicle in snapshot["user_vehicles"]))
        self.assertTrue(any(event.get("event") == "exit" for event in snapshot["events"]))
        self.assertTrue(any(user.get("name") == "Private User" for user in snapshot["users"]))
        self.assertTrue(any(vehicle.get("name") == "Private User" for vehicle in snapshot["user_vehicles"]))

    def test_user_entry_does_not_duplicate_parked_events(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        payload = {
            "name": "Repeat User",
            "vehicle_number": "TN03REPEAT",
            "user_type": "student",
            "vehicle_type": "car",
            "gate": "Gate1",
        }
        runtime.register_user_entry(payload)
        runtime.register_user_entry({"vehicle_number": "TN03REPEAT", "gate": "Gate2"})
        events = [
            event.get("event")
            for event in runtime.get_runtime_snapshot()["events"]
            if event.get("vehicle_number") == "TN03REPEAT"
        ]

        self.assertEqual(events.count("entry"), 1)
        self.assertEqual(events.count("parked"), 1)

    def test_vehicle_event_normalization_removes_repeated_state_rows(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.vehicle_events = [
            {"id": "EV-1", "vehicle_number": "TN05BE8852", "event": "entry", "timestamp": "t1"},
            {"id": "EV-2", "vehicle_number": "TN05BE8852", "event": "parked", "timestamp": "t2"},
            {"id": "EV-3", "vehicle_number": "TN05BE8852", "event": "entry", "timestamp": "t3"},
            {"id": "EV-4", "vehicle_number": "TN05BE8852", "event": "parked", "timestamp": "t4"},
            {"id": "EV-5", "vehicle_number": "TN05BE8852", "event": "parked", "timestamp": "t5"},
            {"id": "EV-6", "vehicle_number": "TN05BE8852", "event": "exit", "timestamp": "t6"},
        ]

        events = [
            event.get("event")
            for event in runtime.get_runtime_snapshot()["events"]
            if event.get("vehicle_number") == "TN05BE8852"
        ]

        self.assertEqual(events, ["entry", "parked", "exit"])

    def test_simulated_redirect_event_is_not_repeated_for_same_step(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        vehicle = {
            "id": 1,
            "number": "SIM-1",
            "type": "car",
            "user_type": "simulated",
            "status": "redirecting",
            "block": "Tech Park",
            "gate": "Gate1",
        }
        runtime._record_vehicle_event(
            "redirect",
            vehicle,
            from_block="MBA Block",
            to_block="Tech Park",
            decision_step=42,
        )
        runtime._record_vehicle_event(
            "redirect",
            vehicle,
            from_block="MBA Block",
            to_block="Tech Park",
            decision_step=42,
        )

        events = [
            event
            for event in runtime.get_runtime_snapshot()["events"]
            if event.get("vehicle_number") == "SIM-1" and event.get("event") == "redirect"
        ]

        self.assertEqual(len(events), 1)

    def test_exit_count_uses_vehicle_status_not_duplicate_history(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.register_user_entry({
            "name": "Exit Count",
            "vehicle_number": "TN04EXIT",
            "user_type": "visitor",
            "vehicle_type": "car",
            "gate": "Gate1",
        })
        runtime.register_user_exit({"vehicle_number": "TN04EXIT"})
        runtime.vehicle_events.append({
            "id": "manual-duplicate",
            "vehicle_number": "TN04EXIT",
            "event": "exit",
            "timestamp": "duplicate",
        })

        stats = runtime.get_runtime_snapshot()["vehicle_stats"]

        self.assertEqual(stats["exited"], 1)
        self.assertEqual(stats["user"], 1)
        self.assertEqual(stats["entering"], 0)
        self.assertEqual(stats["exiting"], 0)

    def test_unknown_exit_returns_user_friendly_not_found(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        response = runtime.register_user_exit({"vehicle_number": "UNKNOWN"})

        self.assertEqual(response["status"], "not_found")
        self.assertEqual(response["notification"], "Vehicle not found")

    def test_snapshot_exposes_agentic_integrity_report(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        runtime.step()
        snapshot = runtime.get_runtime_snapshot()
        integrity = snapshot["agentic_integrity"]

        self.assertIn("score", integrity)
        self.assertIn("checks", integrity)
        self.assertGreaterEqual(integrity["score"], 80)
        self.assertFalse(integrity["issues"]["bad_capacity_blocks"])

    def test_demo_pressure_profiles_change_real_occupancy(self):
        runtime = ParkingRuntimeService(storage_path=self.state_path)
        normal = runtime.apply_demo_pressure("normal")
        heavy = runtime.apply_demo_pressure("near_full")

        def occupancy(snapshot):
            blocks = snapshot["blocks"]
            occupied = sum(block["occupied"] for block in blocks.values())
            capacity = sum(block["capacity"] for block in blocks.values())
            return occupied / capacity

        self.assertLess(occupancy(normal), 0.65)
        self.assertGreater(occupancy(heavy), 0.9)
        self.assertFalse(heavy["agentic_integrity"]["issues"]["bad_capacity_blocks"])

if __name__ == "__main__":
    unittest.main()
