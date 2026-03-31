import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agents import get_model
from app.routing import decide_route

DEFAULT_PROMPTS = [
    "What is the OAuth token expiry and latency requirement for internal services?",
    "Can Tier 1 accounts process cryptocurrency-related transactions?",
    "Walk me through the account opening process for a new retail customer.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real LLM-backed smoke tests for router classification."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to classify. Can be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    model = get_model()

    print("Running router smoke test against the configured model backend.")
    for index, prompt in enumerate(prompts, start=1):
        started_at = time.perf_counter()
        route_name = decide_route(prompt, model)
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        print(f"\n[{index}] Prompt: {prompt}")
        print(f"    Route: {route_name}")
        print(f"    Latency: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
