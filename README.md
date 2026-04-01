# Bank Multi-Agent Expert System

This repository implements a LangGraph-based multi-agent system for querying a large internal banking operations and compliance manual while preserving prefix-cache efficiency.

## Overview

The system contains:

- a lightweight LLM-based router
- three expert agents:
  - Technical Specialist
  - Compliance Auditor
  - Support Concierge
- a FastAPI API with:
  - `POST /chat` for standard responses and cache-metric collection
  - `POST /chat/stream` for streaming responses and TTFT measurement
- durable thread persistence in SQLite

The architecture is designed around one core constraint from the assignment: the operations manual should remain a fixed prompt prefix across expert calls so a serving layer with prefix caching can reuse the expensive prefill work.

## Architecture

### Request Flow

1. The client sends a user message to `/chat` or `/chat/stream`.
2. Prior thread history is loaded from SQLite using `thread_id`.
3. The router performs a small classification call and selects one expert.
4. The selected expert builds a prompt with:
   - the full manual first
   - expert-specific instructions second
   - conversation history last
5. The final answer is returned and the updated thread state is persisted.

### Main Files

- `main.py`: FastAPI app, sync and streaming endpoints
- `app/graph.py`: LangGraph router-to-expert workflow
- `app/agents.py`: router logic, expert invocation, streaming support
- `app/prompts.py`: cache-aware prompt structure
- `app/manual_loader.py`: real or simulated manual loading
- `app/persistence.py`: SQLite thread persistence
- `app/llm_utils.py`: response parsing and metric extraction
- `scripts/cache_benchmark.py`: non-streaming cache benchmark
- `scripts/stream_ttft_benchmark.py`: streaming TTFT benchmark

## Cache Strategy

### High Cache Hit Rate

The prompt is intentionally structured so the large manual is always the first system message for every expert call.
This design assumes a prefix-caching backend (for example LMCache-compatible serving), where identical token prefixes allow KV cache reuse across requests.

That means the shared prefix is:

1. identical manual content
2. identical placement in the message list
3. identical role ordering before any user-specific content appears

Even minor changes in whitespace, formatting, or ordering within the manual prefix would break cache reuse, so the prefix is constructed once and reused verbatim across all agents.

The expert-specific instructions come after the manual. This matters because if the role prompt were placed before the manual, each expert would create a different early-token prefix and reduce cache reuse across agent switches.

In this implementation, the cache-sensitive ordering is:

1. `SystemMessage(manual)`
2. `SystemMessage(expert instructions)`
3. conversation history

This is the key optimization choice in the codebase.

### Dynamic vs Static Content

Static content:

- the full manual
- the prompt wrapper around that manual

Dynamic content:

- expert role instructions
- user messages
- growing thread history

By keeping the manual first, only the suffix changes between expert calls. That gives the serving engine the best chance to reuse the large shared prefix.

## Why This Reduces TTFT

The expensive part of inference for long-context prompts is prefill. If the manual is large and identical across expert calls, the backend can reuse cached KV state for that prefix instead of recomputing it.

In practice this should reduce TTFT most clearly for the second and third expert calls in a sequence, because:

- the first expert call populates the prefix cache
- later expert calls reuse the same manual prefix
- only the later prompt suffix differs

## Measuring Efficiency

This backend exposes useful signals through two different response modes, so the repository uses two measurement paths.

### Non-Streaming: Cache Metrics

Use `POST /chat` or `python scripts/cache_benchmark.py`.

This path is used to inspect fields such as:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `cached_tokens`

This is the best path for proving cache reuse.

### Streaming: TTFT

Use `POST /chat/stream` or `python scripts/stream_ttft_benchmark.py`.

This path is used to measure:

- router latency
- application-observed TTFT
- total stream duration

The streaming endpoint is useful even though the backend does not expose cached-token counters in stream chunks, because TTFT is the most natural latency metric for streamed responses.

### Why The Measurements Are Split

For this provider, the non-streaming response includes cache-usage metadata, while the streaming response emits incremental chunks and ends with `[DONE]` rather than the same final usage object. Because of that, this repository treats:

- non-streaming as the source of cache metrics
- streaming as the source of TTFT metrics

This is intentional and matches the provider behavior instead of trying to infer unsupported metrics from streamed chunks.

The TTFT reported by this repository is measured at the application boundary. It includes client-side and network overhead in addition to model-serving time, so it should be treated as end-to-end observed TTFT rather than raw engine-internal TTFT.

## Benchmark Results

I ran both benchmark scripts in two modes:

- small fallback manual
- larger simulated manual that still fit within the provided endpoint's context window

I initially tried a larger simulation that exceeded the model context limit and received a `400` error indicating a negative `max_tokens` budget. Because of that, the reported "large" run below reflects the largest successful simulated-prefix configuration rather than an arbitrarily repeated manual size.

### Non-Streaming Cache Benchmark

Small fallback manual:

| Expert | Elapsed (ms) | Prompt Tokens | Cached Tokens |
| --- | ---: | ---: | ---: |
| Technical Specialist | 2023.90 | 373 | 372 |
| Compliance Auditor | 949.24 | 375 | 368 |
| Support Concierge | 3047.70 | 377 | 368 |

Larger simulated manual:

| Expert | Elapsed (ms) | Prompt Tokens | Cached Tokens |
| --- | ---: | ---: | ---: |
| Technical Specialist | 3861.89 | 4686 | 288 |
| Compliance Auditor | 1512.13 | 4688 | 4608 |
| Support Concierge | 3134.85 | 4690 | 4608 |

Interpretation:

- With the larger shared prefix, the first expert call paid the prefill cost and showed relatively low cached-token reuse.
- The second and third expert calls showed very high cached-token reuse (`4608` cached prompt tokens), which is the behavior the prompt layout was designed to enable.
- This supports the prefix-alignment strategy: keep the manual fixed and move expert-specific instructions after it.

### Streaming TTFT Benchmark

Small fallback manual:

| Expert | Observed Elapsed (ms) | TTFT (ms) | Total Stream Duration (ms) |
| --- | ---: | ---: | ---: |
| Technical Specialist | 2186.65 | 1909.55 | 2186.45 |
| Compliance Auditor | 2027.59 | 1776.85 | 2027.34 |
| Support Concierge | 3528.24 | 1542.47 | 3528.08 |

Larger simulated manual:

| Expert | Observed Elapsed (ms) | TTFT (ms) | Total Stream Duration (ms) |
| --- | ---: | ---: | ---: |
| Technical Specialist | 2084.27 | 1827.17 | 2084.07 |
| Compliance Auditor | 2022.94 | 1537.09 | 2022.78 |
| Support Concierge | 4046.24 | 1746.96 | 4046.01 |

Interpretation:

- The streaming benchmark measures application-observed TTFT, not raw serving-engine TTFT.
- Despite a significantly larger prompt, later expert calls maintained similar observed TTFT, which is consistent with prefix cache reuse reducing repeated prefill work.
- The support concierge response still had the longest total duration because it produced the longest answer, which affects decode time even when the prompt prefix is reused.
- Cache reuse primarily reduces prefill latency, while decode latency remains dependent on response length.

## State Management

Conversation state is durable across process restarts through SQLite in `app/persistence.py`.

Persisted data includes:

- thread messages
- last selected agent
- update timestamp

The manual is not persisted per thread. It remains shared application state and is reattached during prompt construction. This keeps the large static context outside the conversation store and avoids duplicating the manual into every saved thread.

## Production Readiness

### What Is Implemented

- FastAPI API surface
- input validation with Pydantic
- error logging
- request timing
- durable thread persistence
- offline unit tests
- opt-in real endpoint tests

### Monitoring In Production

If this were deployed in production, I would monitor both application-level and inference-level signals.

Langfuse:

- trace router and expert calls separately
- record route decisions, prompt sizes, TTFT, and total latency
- compare latency by expert and by thread length

vLLM or serving logs:

- inspect prefix-cache hit behavior
- capture cached-token counts
- track prefill vs decode time

Grafana / Prometheus:

- p50/p95 TTFT
- p50/p95 total latency
- cache-hit ratio or cached-token volume
- request rate and error rate
- route distribution by expert

The main operational question is whether later expert calls on the same shared manual show lower prefill cost and improved TTFT.

## Local Development

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Environment Variables

Core endpoint settings:

- `API_BASE_URL`
- `API_KEY`
- `API_USER_ID`
- `MODEL_NAME`

Manual and state settings:

- `MANUAL_PATH`
- `STATE_DB_PATH`

Development vs assignment modes:

- if `MANUAL_PATH` exists, the application loads the real manual file
- otherwise the fallback manual is small by default for fast debugging
- `SIMULATE_LARGE_MANUAL=1` repeats the fallback text to exercise large-prefix behavior when a real manual file is not present
- `SIMULATED_MANUAL_REPEAT_COUNT` controls the simulated size

### Run The API

```bash
uvicorn main:app --reload
```

### Run Tests

Offline tests:

```bash
pytest -q
```

Real endpoint tests:

```bash
RUN_ENDPOINT_TESTS=1 pytest -q -m endpoint
```

## Benchmark Commands

Non-streaming cache metrics:

```bash
python scripts/cache_benchmark.py
```

Streaming TTFT:

```bash
python scripts/stream_ttft_benchmark.py
```

Streaming TTFT with full responses:

```bash
python scripts/stream_ttft_benchmark.py --full-response
```

## Tradeoffs

- I used SQLite for durable thread persistence instead of a LangGraph-native persistent checkpointer because it is simple, reliable, and easy to explain in a take-home setting.
- The router is a separate small LLM call rather than a rule-based classifier, which keeps the implementation close to the multi-agent assignment but adds one extra network hop.
- The fallback manual is small by default to speed development, but assignment validation should use a large simulated manual or a real manual file.
- Long-running conversations would need token-budget management to avoid exceeding the endpoint context window. In a production version, I would preserve the fixed manual prefix, keep recent turns verbatim, and summarize or trim older history before dispatch.

## Final Notes

The core optimization idea in this project is not just “use multiple agents.” It is “use multiple agents without changing the expensive prefix.” The manual stays fixed, expert behavior changes later in the prompt, and the system measures cache reuse and TTFT through the response modes the provider actually supports.
