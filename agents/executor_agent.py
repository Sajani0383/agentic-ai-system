from copy import deepcopy

from adk.trace_logger import trace_logger


class ExecutorAgent:
    def __init__(self, logger=None, max_retries=1):
        self.logger = logger or trace_logger
        self.max_retries = max(0, int(max_retries))
        self.execution_history = []

    def execute(self, review, environment, apply=False):
        action = review.get("revised_action", {"action": "none"})
        approved = bool(review.get("approved", False))

        try:
            if not approved:
                return self._build_noop_result(review, "Critic did not approve execution.")

            validation = self._validate_action(action, environment)
            executable = validation["action"]
            if not validation["valid"]:
                return self._build_failure_result(review, executable, validation["reason"], validation)

            if apply:
                return self._apply_action(review, executable, environment, validation)

            result = self._build_prepared_result(review, executable, validation)
            self._record_execution(result)
            return result
        except Exception as exc:
            result = self._build_failure_result(
                review,
                {"action": "none"},
                f"Executor failed with {type(exc).__name__}: {exc}",
                {"valid": False, "error": str(exc)},
            )
            self._record_execution(result)
            return result

    def execute_sequence(self, reviews, environment, apply=False):
        results = []
        for index, review in enumerate(reviews):
            result = self.execute(review, environment, apply=apply)
            result["sequence_index"] = index
            results.append(result)
            if not result.get("success"):
                break
        return {
            "execution_mode": "direct_apply" if apply else "prepared_sequence",
            "steps_requested": len(reviews),
            "steps_completed": len(results),
            "results": results,
            "success": bool(results and all(result.get("success") for result in results)),
        }

    def get_execution_history(self, limit=20):
        return deepcopy(self.execution_history[-limit:])

    def _validate_action(self, action, environment):
        state = environment.get_state()
        if action.get("action") != "redirect":
            return {
                "valid": True,
                "reason": "No movement required.",
                "action": {"action": "none"},
                "requested": 0,
                "executable_vehicles": 0,
                "partial_execution": False,
            }

        from_zone = action.get("from")
        to_zone = action.get("to")
        requested = max(0, int(action.get("vehicles", 0) or 0))

        if from_zone not in state or to_zone not in state:
            return {
                "valid": False,
                "reason": "Redirect contains an unknown source or destination zone.",
                "action": {"action": "none"},
                "requested": requested,
                "executable_vehicles": 0,
                "partial_execution": False,
            }
        if from_zone == to_zone:
            return {
                "valid": False,
                "reason": "Redirect source and destination cannot be the same zone.",
                "action": {"action": "none"},
                "requested": requested,
                "executable_vehicles": 0,
                "partial_execution": False,
            }
        if requested <= 0:
            return {
                "valid": False,
                "reason": "Redirect must request at least one vehicle.",
                "action": {"action": "none"},
                "requested": requested,
                "executable_vehicles": 0,
                "partial_execution": False,
            }

        source_redirect_capacity = max(
            state[from_zone]["entry"],
            max(0, 12 - state[from_zone]["free_slots"]),
        )
        free_capacity = state[to_zone]["free_slots"]
        executable_vehicles = max(0, min(requested, source_redirect_capacity, free_capacity))
        if executable_vehicles <= 0:
            return {
                "valid": False,
                "reason": "No incoming demand can be rerouted because source pressure or destination capacity is unavailable.",
                "action": {"action": "none"},
                "requested": requested,
                "executable_vehicles": 0,
                "partial_execution": False,
            }

        executable = dict(action)
        executable["vehicles"] = executable_vehicles
        executable["execution_adjusted"] = executable_vehicles != requested
        return {
            "valid": True,
            "reason": "Action is executable.",
            "action": executable,
            "requested": requested,
            "executable_vehicles": executable_vehicles,
            "partial_execution": executable_vehicles < requested,
            "source_available": source_redirect_capacity,
            "destination_capacity": free_capacity,
        }

    def _apply_action(self, review, executable, environment, validation):
        before_state = environment.get_state()
        transfer_report = self._retry_apply(environment, executable)
        after_state = environment.get_state()
        moved = transfer_report.get("moved", 0)
        result = {
            "final_action": executable if moved > 0 else {"action": "none"},
            "execution_note": (
                f"Executor scheduled {moved} incoming arrivals to be rerouted from {transfer_report.get('from')} to {transfer_report.get('to')}."
                if moved > 0
                else "Executor attempted the action but no arrivals could be rerouted."
            ),
            "approved": review.get("approved", False),
            "risk_level": review.get("risk_level", "medium"),
            "success": moved > 0,
            "applied": True,
            "partial_execution": moved < validation.get("requested", 0),
            "requested_vehicles": validation.get("requested", 0),
            "executed_vehicles": moved,
            "transfer_report": transfer_report,
            "state_delta": self._state_delta(before_state, after_state),
            "learning_feedback": self._learning_feedback(review, moved, validation),
        }
        self._record_execution(result)
        return result

    def _retry_apply(self, environment, executable):
        last_report = {"moved": 0, "from": executable.get("from"), "to": executable.get("to"), "requested": executable.get("vehicles", 0)}
        for _attempt in range(self.max_retries + 1):
            last_report = environment.apply_action(executable)
            if last_report.get("moved", 0) > 0:
                return last_report
        return last_report

    def _build_prepared_result(self, review, executable, validation):
        note = (
            f"Executor prepared redirect from {executable.get('from')} to {executable.get('to')} "
            f"for {validation['executable_vehicles']} vehicles."
        )
        if validation["partial_execution"]:
            note += f" Requested {validation['requested']} vehicles, but only {validation['executable_vehicles']} are executable."
        result = {
            "final_action": executable,
            "execution_note": note,
            "approved": review.get("approved", False),
            "risk_level": review.get("risk_level", "medium"),
            "success": True,
            "applied": False,
            "partial_execution": validation["partial_execution"],
            "requested_vehicles": validation["requested"],
            "executed_vehicles": 0,
            "executable_vehicles": validation["executable_vehicles"],
            "validation": validation,
            "learning_feedback": self._learning_feedback(review, validation["executable_vehicles"], validation),
        }
        return result

    def _build_noop_result(self, review, reason):
        result = {
            "final_action": {"action": "none"},
            "execution_note": reason,
            "approved": review.get("approved", False),
            "risk_level": review.get("risk_level", "medium"),
            "success": True,
            "applied": False,
            "partial_execution": False,
            "requested_vehicles": 0,
            "executed_vehicles": 0,
            "executable_vehicles": 0,
            "learning_feedback": {"execution_success": True, "moved": 0, "reason": reason},
        }
        self._record_execution(result)
        return result

    def _build_failure_result(self, review, executable, reason, validation):
        result = {
            "final_action": {"action": "none"},
            "blocked_action": deepcopy(executable),
            "execution_note": reason,
            "approved": review.get("approved", False),
            "risk_level": review.get("risk_level", "medium"),
            "success": False,
            "applied": False,
            "partial_execution": False,
            "requested_vehicles": validation.get("requested", 0),
            "executed_vehicles": 0,
            "executable_vehicles": validation.get("executable_vehicles", 0),
            "validation": validation,
            "learning_feedback": {
                "execution_success": False,
                "moved": 0,
                "reason": reason,
            },
        }
        self._record_execution(result)
        return result

    def _learning_feedback(self, review, moved_or_executable, validation):
        requested = max(1, validation.get("requested", 0))
        return {
            "execution_success": moved_or_executable > 0,
            "fulfillment_ratio": round(moved_or_executable / requested, 3),
            "critic_risk_level": review.get("risk_level", "medium"),
            "critic_risk_score": review.get("risk_score"),
            "partial_execution": validation.get("partial_execution", False),
        }

    def _state_delta(self, before_state, after_state):
        return {
            zone: {
                "occupied_delta": after_state[zone]["occupied"] - before_state[zone]["occupied"],
                "free_delta": after_state[zone]["free_slots"] - before_state[zone]["free_slots"],
            }
            for zone in before_state
            if zone in after_state
        }

    def _record_execution(self, result):
        summary = {
            "success": result.get("success"),
            "applied": result.get("applied"),
            "final_action": result.get("final_action"),
            "requested_vehicles": result.get("requested_vehicles"),
            "executed_vehicles": result.get("executed_vehicles"),
            "note": result.get("execution_note"),
        }
        self.execution_history.append(summary)
        self.execution_history = self.execution_history[-100:]
        level = "INFO" if result.get("success") else "ERROR"
        self.logger.log("-", "executor_execution", summary, level=level)
