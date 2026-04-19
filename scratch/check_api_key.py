import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from llm.client import get_llm, get_llm_status

def test_connection():
    status = get_llm_status()
    print(f"LLM Status: {status['available']}")
    print(f"Message: {status['message']}")
    
    if not status['available']:
        print(f"Last Error: {status['last_error']}")
        return

    llm = get_llm()
    if llm:
        print("Attempting test generation...")
        try:
            response = llm.invoke("Hello, are you there? Reply with 'OK' if you can hear me.")
            print(f"Response: {response.content}")
        except Exception as e:
            print(f"Failed to invoke LLM: {e}")
    else:
        print("Failed to get LLM instance.")

if __name__ == "__main__":
    test_connection()
