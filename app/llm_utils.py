from typing import Any


def extract_content(response) -> str:
    """
    Robustly extracts text content from provider responses.
    Handles plain strings and list-based content blocks.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return " ".join(text_parts)
    return str(content)


def extract_non_stream_response_metrics(response) -> dict[str, Any]:
    """
    Extract metrics from the assignment endpoint's non-streaming response.

    LangChain may normalize token counts into `usage_metadata`, but cached-token
    counters still come from the provider-style fields inside `response_metadata`.
    """
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {})
    prompt_token_details = token_usage.get("prompt_tokens_details", {})

    prompt_tokens = usage_metadata.get("input_tokens") or token_usage.get(
        "prompt_tokens"
    )
    completion_tokens = usage_metadata.get("output_tokens") or token_usage.get(
        "completion_tokens"
    )
    total_tokens = usage_metadata.get("total_tokens") or token_usage.get("total_tokens")

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": prompt_token_details.get("cached_tokens"),
    }
