import json
import os
from copy import deepcopy
from threading import Lock

from agent_controller import AgentController
from agent_memory import AgentMemory
from environment.parking_environment import ParkingEnvironment
from llm_reasoning import get_llm, get_llm_status, get_local_chat_response
from services.mock_notification_service import MockNotificationService


class ParkingRuntimeService:
    def __init__(self, storage_path=None, memory_storage_path=None, notification_storage_path=None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.storage_path = storage_path or os.path.join(base_dir, "memory", "runtime_state.json")
        self.lock = Lock()
        self.environment = ParkingEnvironment()
        self.memory = AgentMemory(storage_path=memory_storage_path)
        self.controller = AgentController(environment=self.environment, memory=self.memory)
        self.notification_service = MockNotificationService(storage_path=notification_storage_path)
        self.latest_result = {}
        self.trace_log = []
        self._load()

    def reset(self, clear_memory=False):
        with self.lock:
            if clear_memory:
                self.environment = ParkingEnvironment()
                self.memory = AgentMemory(storage_path=self.memory.storage_path)
                self.memory.reset(persist=True)
                self.controller = AgentController(environment=self.environment, memory=self.memory)
                self.notification_service.reset()
            else:
                self.controller.reset(clear_memory=False)
            self.latest_result = {}
            self.trace_log = []
            self._save()
            return self.get_runtime_snapshot()

    def set_scenario_mode(self, scenario_mode):
        with self.lock:
            self.environment.set_scenario_mode(scenario_mode)
            self._save()
            return self.get_runtime_snapshot()

    def step(self):
        with self.lock:
            result = self.controller.step()
            notification_dispatch = self.notification_service.dispatch(
                result.get("notifications", []),
                result.get("event_context", {}),
            )
            result["notification_dispatch"] = notification_dispatch
            self.latest_result = result
            self.trace_log.append(
                {
                    "step": result.get("step_number"),
                    "mode": result.get("mode"),
                    "action": deepcopy(result.get("action", {})),
                    "goal": deepcopy(result.get("goal", {})),
                }
            )
            self.trace_log = self.trace_log[-200:]
            self._save()
            return deepcopy(result)

    def ask(self, query):
        with self.lock:
            state = self.environment.get_state()
            latest_result = self.latest_result or {}
            event_context = latest_result.get("event_context", self.environment.get_event_context())
            local_answer = self._answer_operational_query(state, query, latest_result, event_context)
            llm = get_llm()
            if local_answer is not None:
                answer = local_answer
            elif llm is None:
                answer = get_local_chat_response(state, query)
            else:
                try:
                    answer = llm.invoke(
                        f"Parking state: {state}\n"
                        f"Latest result: {latest_result}\n"
                        f"Event context: {event_context}\n"
                        f"Learning profile: {self.memory.get_learning_profile(scenario_mode=self.environment.get_scenario_mode())}\n"
                        f"User question: {query}\n"
                        "Answer in short operational language. If the user asks zone-by-zone details, list every zone explicitly."
                    ).content
                except Exception:
                    answer = get_local_chat_response(state, query)

            response = {
                "query": query,
                "answer": answer,
                "state": state,
                "goal": self.memory.get_active_goal(),
            }
            self.trace_log.append({"query": query, "answer": answer})
            self.trace_log = self.trace_log[-200:]
            self._save()
            return response

    def run_agent_command(self, user_input):
        normalized = user_input.strip().lower()
        if normalized in {"step", "run step", "advance"}:
            result = self.step()
            return {
                "type": "step",
                "message": f"Executed step {result['step_number']} with action {result['action']}.",
                "result": result,
            }
        if normalized in {"reset", "restart"}:
            snapshot = self.reset(clear_memory=False)
            return {
                "type": "reset",
                "message": "Runtime reset completed while keeping persistent historical memory.",
                "result": snapshot,
            }

        return {
            "type": "chat",
            "message": self.ask(user_input)["answer"],
            "result": self.get_runtime_snapshot(),
        }

    def get_runtime_snapshot(self):
        state = self.environment.get_state()
        latest_transition = self.latest_result.get("transition", self.environment.get_last_transition())
        return {
            "state": state,
            "latest_result": deepcopy(self.latest_result),
            "latest_transition": deepcopy(latest_transition),
            "metrics": self.memory.get_metrics(),
            "goal": self.memory.get_active_goal(),
            "scenario_mode": self.environment.get_scenario_mode(),
            "llm_status": get_llm_status(),
            "event_context": deepcopy(latest_transition.get("event_context", self.environment.get_event_context())),
            "notifications": deepcopy(latest_transition.get("notifications", [])),
            "notification_dispatch": self.notification_service.get_recent_deliveries(limit=15),
            "kpis": deepcopy(latest_transition.get("kpis", {})),
            "recent_cycles": self.memory.get_recent_cycles(limit=10),
            "recent_states": self.memory.get_recent_states(limit=10),
            "trace": deepcopy(self.trace_log[-20:]),
        }

    def get_notification_feed(self):
        return self.notification_service.get_recent_deliveries(limit=50)

    def _answer_operational_query(self, state, query, latest_result, event_context):
        query_lower = query.lower()
        if any(
            phrase in query_lower
            for phrase in [
                "occupied in each",
                "all the slots occupied",
                "zone by zone",
                "occupied slots",
            ]
        ):
            lines = [
                f"- {zone}: occupied {data['occupied']}, free {data['free_slots']}, total {data['total_slots']}"
                for zone, data in state.items()
            ]
            return "Occupied slots by zone:\n" + "\n".join(lines)

        if "free slots in each" in query_lower or ("free slots" in query_lower and "each" in query_lower):
            lines = [
                f"- {zone}: free {data['free_slots']}, occupied {data['occupied']}, total {data['total_slots']}"
                for zone, data in state.items()
            ]
            return "Free slots by zone:\n" + "\n".join(lines)

        if "entries and exits" in query_lower or "movement in each" in query_lower:
            lines = [
                f"- {zone}: entries {data['entry']}, exits {data['exit']}, free {data['free_slots']}"
                for zone, data in state.items()
            ]
            return "Current vehicle movement by zone:\n" + "\n".join(lines)

        if "current event" in query_lower or "what event" in query_lower:
            return (
                f"Active event: {event_context.get('name', 'Normal Day')}.\n"
                f"Strategy: {event_context.get('allocation_strategy', '-') }.\n"
                f"Recommended zone: {event_context.get('recommended_zone', '-') }."
            )

        if "latest allocation" in query_lower or "latest decision" in query_lower:
            action = latest_result.get("action", {})
            reason = action.get("reason")
            if not reason:
                critic_notes = latest_result.get("critic_output", {}).get("critic_notes", [])
                reason = critic_notes[0] if critic_notes else "The agent held position because current network pressure does not justify moving vehicles."
            return (
                f"Latest action: {action.get('action', 'none').upper()}.\n"
                f"Route: {action.get('from', '-') } -> {action.get('to', '-') }.\n"
                f"Reason: {reason}."
            )

        return None

    def _load(self):
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return

        environment_snapshot = payload.get("environment")
        memory_snapshot = payload.get("memory")
        if environment_snapshot:
            self.environment.load_snapshot(environment_snapshot)
        if memory_snapshot:
            self.memory.load_export(memory_snapshot)
        self.latest_result = payload.get("latest_result", {})
        self.trace_log = payload.get("trace_log", [])

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        payload = {
            "environment": self.environment.export_snapshot(),
            "memory": self.memory.export(),
            "latest_result": self.latest_result,
            "trace_log": self.trace_log,
        }
        with open(self.storage_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)


runtime_service = ParkingRuntimeService()
