from services.parking_runtime import runtime_service


def run_agent(user_input):
    return runtime_service.run_agent_command(user_input)


def get_trace():
    return runtime_service.get_runtime_snapshot()["trace"]


def get_runtime_snapshot():
    return runtime_service.get_runtime_snapshot()


def get_notification_feed():
    return runtime_service.get_notification_feed()


def step_runtime():
    return runtime_service.step()


def reset_runtime(clear_memory=False):
    return runtime_service.reset(clear_memory=clear_memory)


def set_runtime_scenario(scenario_mode):
    return runtime_service.set_scenario_mode(scenario_mode)
