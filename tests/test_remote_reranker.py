from __future__ import annotations

import json
import traceback

import pytest

httpx = pytest.importorskip("httpx", reason="remote reranker tests require httpx")

from subjective_scoring import (
    CohereRerankerPairScorer,
    RemoteRerankerRequestError,
    RemoteRerankerResponseError,
    SubjectiveScoringService,
)


URL = "https://router.example.test/v1/rerank"
API_KEY = "secret-test-key"
MODEL = "Pro/BAAI/bge-reranker-v2-m3"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _scorer(handler, **kwargs) -> CohereRerankerPairScorer:
    return CohereRerankerPairScorer(
        url=URL,
        api_key=API_KEY,
        model=MODEL,
        client=_client(handler),
        **kwargs,
    )


def test_empty_pairs_skip_http_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty input must not perform an HTTP request")

    assert _scorer(handler).score_pairs([]) == []


def test_batches_documents_and_restores_ranked_result_order():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert request.headers["content-type"] == "application/json"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    scorer = _scorer(
        handler,
        max_chunks_per_doc=16,
        overlap_tokens=32,
    )
    scores = scorer.score_pairs([("student", "point-a"), ("student", "point-b")])

    assert scores == [0.2, 0.9]
    assert requests == [
        {
            "model": MODEL,
            "query": "student",
            "documents": ["point-a", "point-b"],
            "top_n": 2,
            "return_documents": False,
            "max_chunks_per_doc": 16,
            "overlap_tokens": 32,
        }
    ]


def test_groups_distinct_queries_and_preserves_pair_order():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        queries.append(payload["query"])
        base = 0.1 if payload["query"] == "q1" else 0.7
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": base + index * 0.1}
                    for index in range(len(payload["documents"]))
                ]
            },
        )

    scores = _scorer(handler).score_pairs(
        [("q1", "a"), ("q2", "b"), ("q1", "c")]
    )

    assert queries == ["q1", "q2"]
    assert scores == pytest.approx([0.1, 0.7, 0.2])


def test_missing_results_default_to_zero_and_scores_are_clamped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 1.4},
                    {"index": 2, "relevance_score": -0.3},
                ]
            },
        )

    scores = _scorer(handler).score_pairs(
        [("q", "a"), ("q", "b"), ("q", "c")]
    )
    assert scores == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "results, message",
    [
        (
            [
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.6},
            ],
            "duplicate",
        ),
        ([{"index": 2, "relevance_score": 0.5}], "out of range"),
        ([{"index": "0", "relevance_score": 0.5}], "index"),
        ([{"index": 0, "relevance_score": "bad"}], "score"),
    ],
)
def test_rejects_malformed_result_entries(results, message):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results})

    with pytest.raises(RemoteRerankerResponseError, match=message):
        _scorer(handler).score_pairs([("q", "a")])


def test_rejects_non_finite_score():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"results":[{"index":0,"relevance_score":NaN}]}',
            headers={"content-type": "application/json"},
        )

    with pytest.raises(RemoteRerankerResponseError, match="finite"):
        _scorer(handler).score_pairs([("q", "a")])


@pytest.mark.parametrize("payload", [{}, {"results": {}}, {"results": ["bad"]}])
def test_rejects_malformed_response_shape(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(RemoteRerankerResponseError):
        _scorer(handler).score_pairs([("q", "a")])


def test_rejects_invalid_json_without_exposing_content_or_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"student-secret-response")

    with pytest.raises(RemoteRerankerResponseError) as caught:
        _scorer(handler).score_pairs([("student-secret", "document-secret")])

    message = str(caught.value)
    assert API_KEY not in message
    assert "student-secret" not in message
    assert "document-secret" not in message


def test_http_error_is_sanitized():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"document-secret")

    with pytest.raises(RemoteRerankerRequestError, match="503") as caught:
        _scorer(handler, max_retries=0).score_pairs(
            [("student-secret", "document-secret")]
        )

    message = str(caught.value)
    assert API_KEY not in message
    assert "student-secret" not in message
    assert "document-secret" not in message


def test_transport_timeout_is_sanitized():
    sensitive_query = "-".join(["student", "secret"])
    sensitive_document = "-".join(["document", "secret"])

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(sensitive_query, request=request)

    with pytest.raises(RemoteRerankerRequestError) as caught:
        _scorer(handler, max_retries=0).score_pairs(
            [(sensitive_query, sensitive_document)]
        )

    message = str(caught.value)
    assert API_KEY not in message
    assert sensitive_query not in message
    assert sensitive_document not in message
    rendered_traceback = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert API_KEY not in rendered_traceback
    assert sensitive_query not in rendered_traceback
    assert sensitive_document not in rendered_traceback


def test_retries_429_once_and_honors_numeric_retry_after():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.8}]},
        )

    scores = _scorer(handler, sleep_fn=sleeps.append).score_pairs([("q", "d")])

    assert scores == [0.8]
    assert calls == 2
    assert sleeps == [3.0]


def test_retries_503_with_exponential_backoff_until_success():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.6}]},
        )

    scores = _scorer(handler, sleep_fn=sleeps.append).score_pairs([("q", "d")])

    assert scores == [0.6]
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_does_not_retry_non_retryable_4xx():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, content=b"document-secret")

    with pytest.raises(RemoteRerankerRequestError, match="400"):
        _scorer(handler, sleep_fn=sleeps.append).score_pairs(
            [("student-secret", "document-secret")]
        )

    assert calls == 1
    assert sleeps == []


def test_retries_connection_failures_then_raises_sanitized_error():
    calls = 0
    sleeps: list[float] = []
    sensitive_query = "student-secret"
    sensitive_document = "document-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(sensitive_query, request=request)

    with pytest.raises(RemoteRerankerRequestError) as caught:
        _scorer(handler, sleep_fn=sleeps.append).score_pairs(
            [(sensitive_query, sensitive_document)]
        )

    assert calls == 3
    assert sleeps == [1.0, 2.0]
    message = str(caught.value)
    assert API_KEY not in message
    assert sensitive_query not in message
    assert sensitive_document not in message


def test_does_not_retry_malformed_success_response():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": {}})

    with pytest.raises(RemoteRerankerResponseError):
        _scorer(handler, sleep_fn=lambda seconds: None).score_pairs([("q", "d")])

    assert calls == 1


def test_repr_does_not_expose_api_key():
    scorer = _scorer(lambda request: httpx.Response(200, json={"results": []}))
    rendered = repr(scorer)
    assert API_KEY not in rendered
    assert URL in rendered
    assert MODEL in rendered


def test_close_does_not_close_injected_client():
    client = _client(lambda request: httpx.Response(200, json={"results": []}))
    scorer = CohereRerankerPairScorer(
        url=URL,
        api_key=API_KEY,
        model=MODEL,
        client=client,
    )

    scorer.close()

    assert client.is_closed is False
    client.close()


def test_context_manager_closes_owned_client(monkeypatch):
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    scorer = CohereRerankerPairScorer(
        url=URL,
        api_key=API_KEY,
        model=MODEL,
    )
    client = scorer._client

    with scorer as active:
        assert active is scorer
        assert client.is_closed is False

    assert client.is_closed is True


def test_remote_pair_scorer_integrates_with_subjective_service():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        relevance = 1.0 if payload["query"] == "point one" else 0.02
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": relevance}
                    for index in range(len(payload["documents"]))
                ]
            },
        )

    scorer = _scorer(handler)
    service = SubjectiveScoringService(
        allow_model_load=False,
        text_pair_scorer=scorer,
    )
    result = service.score(
        {
            "question_id": "q1",
            "max_score": 10,
            "scoring_mode": "text",
            "student_answer": "student answer",
            "scoring_points": [
                {"id": "p1", "text": "point one", "score": 5},
                {"id": "p2", "text": "point two", "score": 5},
            ],
        }
    )

    assert len(calls) == 2
    assert {call["query"] for call in calls} == {"point one", "point two"}
    assert result.score == 4.2
    assert result.track == "TextRerankerScorer"
    assert "原始相关度 1.00；校准覆盖度 0.85" in result.matched_points[0].reason


def test_text_evidence_batching_is_bounded_by_point_count_and_caches_reference():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": 0.2}
                    for index in range(len(payload["documents"]))
                ]
            },
        )

    service = SubjectiveScoringService(
        allow_model_load=False,
        text_pair_scorer=_scorer(handler),
    )
    request = {
        "question_id": "perf",
        "max_score": 10,
        "scoring_mode": "text",
        "reference_answer": "资源使用 URI 标识。请求保持无状态。",
        "student_answer": "用 URI 标识资源，每个请求都应自包含。",
        "scoring_points": [
            {
                "id": "resource",
                "text": "资源使用 URI 标识",
                "score": 5,
                "required": True,
            },
            {
                "id": "stateless",
                "text": "请求保持无状态",
                "score": 5,
                "required": True,
            },
        ],
    }

    service.score(request)
    assert len(calls) == 2
    assert all(call["documents"] for call in calls)

    service.score(request)
    assert len(calls) == 4
    assert {call["query"] for call in calls} == {
        "资源使用 URI 标识",
        "请求保持无状态",
    }


def test_remote_pair_scorer_integrates_with_code_scoring():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.8}]},
        )

    scorer = _scorer(handler)
    service = SubjectiveScoringService(
        allow_model_load=False,
        code_pair_scorer=scorer,
    )
    result = service.score(
        {
            "question_id": "c1",
            "max_score": 10,
            "scoring_mode": "code",
            "code_language": "python",
            "reference_answer": "def add(a, b):\n    return a + b\n",
            "student_answer": "def add(x, y):\n    return x + y\n",
        }
    )

    assert calls == 1
    assert result.track == "CodeHybridScorer"
    assert result.score > 0
