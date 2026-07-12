# Cohere-Compatible Cloud Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable, secure Cohere-compatible HTTP reranker that implements `PairScorer` without requiring local model weights.

**Architecture:** A new `rerankers` package contains a synchronous `CohereRerankerPairScorer` and focused exception hierarchy. It batches documents by query, calls a configurable `/rerank` endpoint through an injectable `httpx.Client`, validates and reorders results, and integrates with the existing text/code scorer injection points. HTTP support remains optional through a `remote` extra.

**Tech Stack:** Python 3.12+, httpx, pytest, uv

## Global Constraints

- Never log or expose API keys, authorization headers, submitted queries, or documents in exception text.
- Do not contact a real remote API in tests.
- Preserve input score order even when remote results are ranked.
- Do not silently fall back to lexical similarity after a remote failure.
- Keep `httpx` optional for local-only users.

---

### Task 1: Define Public Remote Reranker Behavior

**Files:**
- Create: `tests/test_remote_reranker.py`

**Interfaces:**
- Consumes: future `subjective_scoring.CohereRerankerPairScorer`
- Produces: executable contract for batching, ordering, validation, security, and scorer injection

- [ ] **Step 1: Add failing tests for batching and ordering**

Create `httpx.MockTransport` handlers that assert the bearer header and request JSON, then return ranked `results`. Verify empty input performs no request, one query produces one request, multiple queries produce one request per query, and returned scores follow original pair order.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `env PYTHONPATH=src /Users/yhw/Code/Github/subjective-scoring/.venv/bin/pytest tests/test_remote_reranker.py -q`

Expected: collection fails because `CohereRerankerPairScorer` is not exported.

- [ ] **Step 3: Add failing validation and security tests**

Cover missing results mapping to `0.0`, score clamping, duplicate indexes, out-of-range indexes, non-finite scores, malformed JSON, HTTP status errors, transport timeouts, sanitized `repr`, and sanitized exception messages.

- [ ] **Step 4: Add a failing service-injection test**

Inject the remote scorer through `SubjectiveScoringService(text_pair_scorer=..., allow_model_load=False)` and verify a text scoring request receives remote similarity scores without local model loading.

---

### Task 2: Implement The Cohere-Compatible Adapter

**Files:**
- Create: `src/subjective_scoring/rerankers/__init__.py`
- Create: `src/subjective_scoring/rerankers/cohere.py`
- Modify: `src/subjective_scoring/__init__.py`

**Interfaces:**
- Produces: `CohereRerankerPairScorer`, `RemoteRerankerError`, `RemoteRerankerRequestError`, `RemoteRerankerResponseError`

- [ ] **Step 1: Implement exceptions and optional HTTP loading**

Define the three public exceptions. Import `httpx` lazily during adapter construction and raise an actionable `ImportError` directing users to `subjective-scoring[remote]` when unavailable.

- [ ] **Step 2: Implement construction and safe lifecycle**

Accept `url`, `api_key`, `model`, `timeout`, optional `max_chunks_per_doc`, optional `overlap_tokens`, and optional injected client. Validate required strings, retain whether the adapter owns the client, implement `close()`, `__enter__`, `__exit__`, and a `repr` containing only URL and model.

- [ ] **Step 3: Implement grouped requests and result mapping**

Group input pairs by query, send `model`, `query`, `documents`, `top_n=len(documents)`, and `return_documents=False`, then map remote indexes back to the original pair positions. Include optional chunk fields only when configured.

- [ ] **Step 4: Implement response validation and safe errors**

Reject non-success responses, invalid JSON, non-list results, non-object entries, non-integer/duplicate/out-of-range indexes, and non-numeric or non-finite scores. Clamp finite scores into `[0.0, 1.0]`; leave missing indexes as `0.0`. Do not include response bodies or submitted content in raised messages.

- [ ] **Step 5: Export the public API**

Re-export the adapter and exceptions from `subjective_scoring.rerankers` and top-level `subjective_scoring`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `env PYTHONPATH=src /Users/yhw/Code/Github/subjective-scoring/.venv/bin/pytest tests/test_remote_reranker.py -q`

Expected: all remote reranker tests pass.

---

### Task 3: Add Optional Dependency And Documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

**Interfaces:**
- Produces: `remote` installation extra and documented application configuration

- [ ] **Step 1: Add the remote extra**

Add `remote = ["httpx>=0.27"]`. Include `remote` in the `all` and `dev` self-referential extras without moving `httpx` into base dependencies.

- [ ] **Step 2: Regenerate the lock file**

Run: `uv lock`.

Expected: lock metadata records the new remote extra and httpx dependency.

- [ ] **Step 3: Document cloud reranking**

Add installation and usage examples showing `subjective-scoring[remote]`, environment-provided URL/key/model values, `allow_model_load=False`, and injection through both `text_pair_scorer` and `code_pair_scorer`. State that secrets belong to the application environment.

- [ ] **Step 4: Run tests without a real API**

Run: `env PYTHONPATH=src /Users/yhw/Code/Github/subjective-scoring/.venv/bin/pytest -q`

Expected: the full suite passes.

---

### Task 4: Verify Packaging And Security Boundaries

**Files:**
- Inspect: all changed files

**Interfaces:**
- Produces: release-ready evidence for the new optional capability

- [ ] **Step 1: Verify package import and exports**

Run a Python command importing the adapter and exceptions from `subjective_scoring` and `subjective_scoring.rerankers`.

- [ ] **Step 2: Verify no secrets or live calls exist**

Search changed source, tests, and documentation for real API keys and confirm every HTTP test uses `MockTransport`.

- [ ] **Step 3: Run final checks**

Run `git diff --check`, the focused remote tests, and the complete test suite.

- [ ] **Step 4: Review the final diff**

Confirm the change is limited to the reranker package, exports, optional dependency metadata, tests, and README documentation.
