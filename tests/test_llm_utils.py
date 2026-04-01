from types import SimpleNamespace

from app.llm_utils import extract_non_stream_response_metrics


def test_extract_non_stream_response_metrics_from_response_metadata():
    response = SimpleNamespace(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        },
        usage_metadata={},
        additional_kwargs={},
    )

    metrics = extract_non_stream_response_metrics(response)

    assert metrics["prompt_tokens"] == 120
    assert metrics["completion_tokens"] == 30
    assert metrics["total_tokens"] == 150
    assert metrics["cached_tokens"] == 100


def test_extract_non_stream_response_metrics_from_usage_metadata():
    response = SimpleNamespace(
        response_metadata={},
        usage_metadata={
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250,
        },
        additional_kwargs={},
    )

    metrics = extract_non_stream_response_metrics(response)

    assert metrics["prompt_tokens"] == 200
    assert metrics["completion_tokens"] == 50
    assert metrics["total_tokens"] == 250
    assert metrics["cached_tokens"] is None
