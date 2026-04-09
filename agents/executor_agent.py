class ExecutorAgent:
    def execute(self, review, environment):
        action = review.get("revised_action", {"action": "none"})
        executable = action if review.get("approved") else {"action": "none"}

        if executable.get("action") == "redirect":
            from_zone = executable.get("from")
            to_zone = executable.get("to")
            capacity = environment.get_state().get(to_zone, {}).get("free_slots", 0)
            note = (
                f"Executor prepared redirect from {from_zone} to {to_zone} "
                f"for up to {min(executable.get('vehicles', 0), capacity)} vehicles."
            )
        else:
            note = "Executor chose not to move vehicles this step."

        return {
            "final_action": executable,
            "execution_note": note,
            "approved": review.get("approved", False),
            "risk_level": review.get("risk_level", "medium"),
        }
