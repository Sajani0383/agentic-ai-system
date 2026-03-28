from agent_memory import AgentMemory
from environment.parking_environment import ParkingEnvironment
from llm_reasoning import create_llm_agent
from tools import get_tools

trace_log = []


def run_agent(user_input):
    environment = ParkingEnvironment()
    history = AgentMemory()
    tools = get_tools(environment, history)
    agent = create_llm_agent(tools)
    result = agent.run(user_input)
    trace_log.append({"input": user_input, "output": result})
    return result


def get_trace():
    return trace_log
