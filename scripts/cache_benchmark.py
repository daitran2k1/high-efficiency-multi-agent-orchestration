import sys
import time
from pathlib import Path

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


def main() -> None:
    print("Running non-streaming cache benchmark with a shared manual prefix.\n")
    for route_name, prompt in BENCHMARK_CASES:
        started_at = time.perf_counter()
        result = EXPERTS[route_name].invoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        metrics = result["metrics"]

        print(f"[{route_name}]")
        print(f"Prompt: {prompt}")
        print(f"Elapsed: {elapsed_ms:.2f} ms")
        print(f"Prompt tokens: {metrics.get('prompt_tokens')}")
        print(f"Completion tokens: {metrics.get('completion_tokens')}")
        print(f"Total tokens: {metrics.get('total_tokens')}")
        print(f"Cached tokens: {metrics.get('cached_tokens')}")
        print(f"Response preview: {result['content'][:120]}")
        print()


if __name__ == "__main__":
    main()
