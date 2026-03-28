from agent_controller import AgentController
from environment.parking_environment import ParkingEnvironment
from llm_reasoning import create_llm_agent
from tools import get_tools


def run_simulation(steps=10):
    print("SMART PARKING AGENTIC AI SYSTEM")

    environment = ParkingEnvironment()
    controller = AgentController(environment=environment, use_logger=False)

    print("Parking Zones:", environment.zones)

    tools = get_tools(environment, controller.memory)
    llm_agent = create_llm_agent(tools)
    overview = llm_agent.invoke(
        "Find the best parking zone based on availability, demand and congestion."
    )
    print("\nStrategic Overview")
    print(overview["output"])

    print("\n--- SIMULATION START ---")

    controller.reset()

    for step in range(steps):
        print(f"\nSTEP: {step + 1}")

        result = controller.step()
        new_state = result["state"]

        print("Chosen Action:", result["action"])
        print("Mode:", result["mode"])
        print("Environment Reward:", result["environment_reward"])
        print("Reward Agent Score:", result["reward_score"])
        print(
            "Best Zone:",
            max(new_state, key=lambda zone: new_state[zone]["free_slots"]),
        )

    print("\n--- SIMULATION END ---")


if __name__ == "__main__":
    run_simulation()
