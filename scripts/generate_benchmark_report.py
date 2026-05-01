import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.parking_runtime import ParkingRuntimeService


def main():
    parser = argparse.ArgumentParser(description="Generate SRM agentic parking benchmark report files.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    runtime = ParkingRuntimeService(storage_path="memory/runtime_state.json")
    runtime.run_benchmark(episodes=args.episodes, steps_per_episode=args.steps)
    result = runtime.export_benchmark_report(output_dir=args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k.endswith("_path")}, indent=2))


if __name__ == "__main__":
    main()
