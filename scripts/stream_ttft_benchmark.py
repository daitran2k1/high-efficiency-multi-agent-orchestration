import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agents import EXPERTS

BENCHMARK_CASES = [
    (
        "technical_specialist",
        "What is the OAuth token expiry and latency requirement for internal services?",
    ),
    (
        "compliance_auditor",
        "Can Tier 1 accounts process cryptocurrency-related transactions?",
    ),
    (
        "support_concierge",
        "Explain the account opening process for a new retail customer in simple steps.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run streaming TTFT benchmarks against the configured model backend."
    )
    parser.add_argument(
        "--full-response",
        action="store_true",
        help="Print the full streamed response instead of a short preview.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    print("Running streaming TTFT benchmark with a shared manual prefix.\n")
    for route_name, prompt in BENCHMARK_CASES:
        started_at = time.perf_counter()
        final_metrics: dict[str, Any] | None = None
        response_parts = []

        async for item in EXPERTS[route_name].astream(
            {"messages": [HumanMessage(content=prompt)]}
        ):
            if item["type"] == "token":
                response_parts.append(item["content"])
            elif item["type"] == "done":
                final_metrics = item["metrics"]

        if final_metrics is None:
            raise RuntimeError(
                f"Stream completed without final metrics for {route_name}"
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        full_response = "".join(response_parts)
        rendered_response = full_response if args.full_response else full_response[:120]
        response_label = "Response" if args.full_response else "Response preview"

        print(f"[{route_name}]")
        print(f"Prompt: {prompt}")
        print(f"Observed elapsed: {elapsed_ms:.2f} ms")
        print(f"TTFT: {final_metrics.get('ttft_ms')}")
        print(f"Total stream duration: {final_metrics.get('total_stream_duration_ms')}")
        print(f"Streamed characters: {final_metrics.get('streamed_characters')}")
        print(f"{response_label}: {rendered_response}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
