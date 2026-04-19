import argparse
import logging
import sys

from ui.adk_dashboard import main


def setup_cli_and_logging():
    """Sets up standard command line interfaces and global logging parameters."""
    parser = argparse.ArgumentParser(description="SRM Agentic Parking Command Center Dashboard Loader")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging mode")
    parser.add_argument("--config", type=str, help="Path to custom configuration file", default="")

    # parse_known_args avoids crashing if Streamlit injects extra unseen CLI flags internally
    args, _ = parser.parse_known_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting up SRM Agentic Parking Command Center Dashboard...")
    if args.debug:
        logging.debug(f"Debug mode activated. Custom config profile: {args.config if args.config else 'None'}")

    return args


def bootstrap():
    """Safely bootstraps the environment before launching the Main UI application loop."""
    setup_cli_and_logging()
    try:
        main()
    except Exception as e:
        logging.error(f"Dashboard failed to launch or crashed unexpectedly: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: Dashboard failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    bootstrap()
