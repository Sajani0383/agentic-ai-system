import streamlit as st
import logging
from services.parking_runtime import runtime_service

class BackendBridge:
    """Safe abstraction layer over the raw Runtime Service. Prevents Streamlit crashing."""
    
    @staticmethod
    def get_snapshot():
        try:
            return runtime_service.get_runtime_snapshot()
        except Exception as e:
            logging.error(f"Backend Bridge crashed accessing snapshot: {e}")
            st.error("Lost connection to the simulation core. Operating in offline safe-mode.")
            return {}

    @staticmethod
    def set_scenario(mode: str):
        try:
            runtime_service.set_scenario_mode(mode)
            return True
        except Exception as e:
            st.error(f"Failed to change scenario: {e}")
            return False

    @staticmethod
    def set_llm_mode(mode: str):
        try:
            runtime_service.set_llm_mode(mode)
            return True
        except Exception as e:
            st.error(f"Failed to change LLM mode: {e}")
            return False

    @staticmethod
    def set_force_llm(enabled: bool):
        try:
            runtime_service.set_force_llm(enabled)
            return True
        except Exception as e:
            st.error(f"Failed to toggle Strategic Overdrive: {e}")
            return False

    @staticmethod
    def reset_llm_runtime_state():
        try:
            runtime_service.reset_llm_runtime_state()
            return True
        except Exception as e:
            st.error(f"Failed to reset AI quota state: {e}")
            return False

    @staticmethod
    def step_simulation():
        try:
            runtime_service.step()
            return True
        except Exception as e:
            st.error(f"Simulation tick failed: {e}")
            return False

    @staticmethod
    def reset(clear_memory=False):
        try:
            runtime_service.reset(clear_memory=clear_memory)
            return True
        except Exception as e:
            st.error(f"Failed to reset environment: {e}")
            return False

    @staticmethod
    def run_benchmark(episodes=3, steps=10):
        try:
            runtime_service.run_benchmark(episodes=episodes, steps_per_episode=steps)
            return True
        except Exception as e:
            st.error(f"Benchmark run failed: {e}")
            return False

    @staticmethod
    def ask(query: str):
        try:
            return runtime_service.ask(query)
        except Exception as e:
            return {"answer": f"API Error: {e}"}

api_bridge = BackendBridge()
