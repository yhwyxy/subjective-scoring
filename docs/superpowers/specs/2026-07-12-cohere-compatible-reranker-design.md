# Cohere-Compatible Cloud Reranker Design

## Goal

Add a reusable synchronous cloud reranker to `subjective-scoring` that implements the existing `PairScorer` protocol and works with Cohere-compatible `/rerank` APIs, including `https://router.tumuer.me/v1/rerank`.

The library provides the HTTP capability. Applications such as `examSystem` remain responsible for supplying the endpoint URL, API key, model ID, and deployment policy.

## Public API

The library will export `CohereRerankerPairScorer` from the package root:

```python
from subjective_scoring import CohereRerankerPairScorer

reranker = CohereRerankerPairScorer(
    url="https://router.tumuer.me/v1/rerank",
    api_key="...",
    model="Pro/BAAI/bge-reranker-v2-m3",
    timeout=30.0,
)
```

The class implements:

```python
def score_pairs(
    self,
    pairs: Sequence[tuple[str, str]],
) -> list[float]:
    ...
```

It can be injected into `SubjectiveScoringService` through `text_pair_scorer` and `code_pair_scorer` while `allow_model_load=False` prevents local model downloads.

## Protocol Mapping

For each distinct query in `pairs`, the adapter sends one request:

```json
{
  "model": "Pro/BAAI/bge-reranker-v2-m3",
  "query": "student answer",
  "documents": ["point one", "point two"],
  "top_n": 2,
  "return_documents": false
}
```

Authentication uses `Authorization: Bearer <api_key>` and JSON content type. Optional `max_chunks_per_doc` and `overlap_tokens` values are included only when configured.

Responses use the Cohere-compatible shape:

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91},
    {"index": 1, "relevance_score": 0.34}
  ]
}
```

Results may be ranked rather than returned in document order. The adapter restores the original pair order using `index`. Missing results receive `0.0`; duplicate or out-of-range indexes are rejected as malformed responses. Scores must be finite numbers and are clamped to `[0.0, 1.0]`.

Although current text and code scorers submit a single query per `score_pairs` call, the adapter groups multiple distinct queries and preserves the complete input order so it remains a valid general `PairScorer` implementation.

## HTTP Client And Dependencies

HTTP calls use `httpx.Client`. The constructor accepts an optional injected client for connection reuse and deterministic tests. When no client is provided, the adapter owns a client and supports `close()` plus context-manager cleanup.

`httpx>=0.27` will be added under a `remote` optional dependency. Importing the main package without `subjective-scoring[remote]` must continue to work. Instantiating the cloud adapter without `httpx` raises an actionable dependency error.

The `all` and `dev` extras will include `remote`. Local-only installations remain free of HTTP dependencies.

## Error Handling And Security

The adapter raises library-specific exceptions for:

- HTTP/network and timeout failures
- non-success HTTP status codes
- invalid JSON
- missing or malformed `results`
- invalid result indexes or relevance scores

Exception messages identify the operation and status without including the API key or authorization header. The API key is stored privately and excluded from `repr` output. Response bodies are not copied wholesale into exceptions because providers may echo submitted exam content.

The adapter does not silently fall back to lexical scoring. The surrounding scoring pipeline already converts scorer failures into a zero machine score with manual review, making remote failure explicit instead of producing a potentially misleading grade.

## Files And Exports

New package structure:

```text
src/subjective_scoring/rerankers/
  __init__.py
  cohere.py
```

`CohereRerankerPairScorer` and its public exception types are re-exported from `subjective_scoring.rerankers` and the top-level `subjective_scoring` package.

## Testing

Tests use `httpx.MockTransport` and never contact a real provider. Coverage includes:

1. Empty input returns an empty list without an HTTP request.
2. A single query is sent as one batched request with the expected headers and payload.
3. Ranked response results are restored to original document order.
4. Multiple distinct queries are grouped into separate requests while preserving pair order.
5. Missing results map to `0.0`.
6. Scores are clamped to `[0.0, 1.0]`.
7. Duplicate indexes, out-of-range indexes, non-finite scores, malformed JSON, HTTP errors, and timeouts raise the appropriate exception.
8. Error text and object representation do not expose the API key or submitted document content.
9. The adapter can be injected into text and code scoring without attempting to load a local CrossEncoder.

The complete existing test suite must remain green with and without the `remote` extra installed.

## Application Integration

`examSystem` will configure the adapter from environment variables after a new `subjective-scoring` tag is published. No endpoint, API key, or model ID is committed to this library or to application source code.
