"""LLM 判分分支单元测试：全部使用 httpx.MockTransport，不打真实 API。"""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx", reason="LLM judge tests require httpx")

from subjective_scoring import (
    JudgeBackend,
    LLMJudgeClient,
    LLMJudgeConfig,
    LLMJudgeRequestError,
    LLMJudgeResponseError,
    LLMJudgeScorer,
    ScoringMode,
    ScoringRequest,
    SubjectiveScoringService,
)


URL = "https://router.tumuer.me/v1"
API_KEY = "sk-secret-test-key"
MODEL = "deepseek-chat"


def _config(**kwargs) -> LLMJudgeConfig:
    defaults = {"max_retries": 0}
    defaults.update(kwargs)
    return LLMJudgeConfig(
        url=URL,
        api_key=API_KEY,
        model=MODEL,
        **defaults,
    )


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    """本机代理指向 SOCKS 时会触发 httpx 缺 socksio，统一屏蔽代理环境变量。"""
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def _client(handler) -> LLMJudgeClient:
    transport = httpx.MockTransport(handler)
    return LLMJudgeClient(
        config=_config(),
        client=httpx.Client(transport=transport, timeout=5),
        sleep_fn=lambda seconds: None,
    )


def _scorer(handler=None, **kwargs) -> LLMJudgeScorer:
    client = _client(handler) if handler is not None else None
    return LLMJudgeScorer(config=_config(), client=client, **kwargs)


def _ok_response(points=None, notes=None) -> dict:
    body: dict = {
        "points": points
        or [
            {
                "point_id": "p1",
                "score": 5.0,
                "relation": "supported",
                "confidence": 0.95,
                "evidence": "索引可以提高查询效率",
                "reason": "覆盖评分点",
            }
        ],
    }
    if notes:
        body["notes"] = notes
    return {"choices": [{"message": {"content": json.dumps(body)}}]}


def _request(**overrides) -> ScoringRequest:
    base = {
        "question_id": "q1",
        "max_score": 10,
        "scoring_mode": "text",
        "question": "索引有什么作用？",
        "reference_answer": "索引可以提高查询效率。",
        "student_answer": "索引可以提高查询效率，加快查询速度。",
        "scoring_points": [
            {"id": "p1", "text": "索引提高查询效率", "score": 5},
            {"id": "p2", "text": "加快查询速度", "score": 5},
        ],
    }
    base.update(overrides)
    return ScoringRequest.model_validate(base)


# ---------------------------------------------------------------------------
# 1. Prompt 与请求头 / payload
# ---------------------------------------------------------------------------


def test_prompt_includes_question_points_reference_and_student():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert request.headers["content-type"] == "application/json"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    scorer = _scorer(handler)
    scorer.score(_request())

    payload = requests[0]
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.0
    assert payload["response_format"] == {"type": "json_object"}
    messages = payload["messages"]
    user = messages[1]["content"]
    assert "索引有什么作用？" in user
    assert "索引提高查询效率" in user
    assert "索引可以提高查询效率。" in user
    assert "索引可以提高查询效率，加快查询速度。" in user


def test_prompt_notes_code_language():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    scorer = _scorer(handler)
    scorer.score(_request(scoring_mode="code", code_language="python"))
    user = requests[0]["messages"][1]["content"]
    assert "python" in user


def test_prompt_includes_calculation_config():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    scorer = _scorer(handler)
    scorer.score(
        _request(
            scoring_mode="calculation",
            scoring_points=[],
            scoring_config={
                "calculation": {
                    "steps": [
                        {
                            "id": "s1",
                            "description": "计算电阻",
                            "expected": 5.0,
                            "score": 5,
                            "unit": "Ω",
                        }
                    ]
                }
            },
        )
    )
    user = requests[0]["messages"][1]["content"]
    assert "计算电阻" in user
    assert "期望值 5.0 Ω" in user


# ---------------------------------------------------------------------------
# 2. 合法响应映射 matched / missed / confidence / track
# ---------------------------------------------------------------------------


def test_valid_response_maps_matched_and_confidence():
    scorer = _scorer(
        lambda request: httpx.Response(
            200,
            json=_ok_response(
                points=[
                    {
                        "point_id": "p1",
                        "score": 5.0,
                        "relation": "supported",
                        "confidence": 0.95,
                        "evidence": "索引可以提高查询效率",
                    },
                    {
                        "point_id": "p2",
                        "score": 2.0,
                        "relation": "unknown",
                        "confidence": 0.6,
                    },
                ]
            ),
        )
    )
    result = scorer.score(_request())

    assert result.scorer == "LLMJudgeScorer"
    assert result.score == 7.0
    # 加权平均 = (0.95*5 + 0.6*5) / 10 = 0.775
    assert result.confidence == pytest.approx(0.775, abs=1e-4)
    assert result.max_score == 10.0
    assert [item.point_id for item in result.matched_evidence] == ["p1"]
    assert [item.point_id for item in result.missed_evidence] == ["p2"]
    assert result.matched_evidence[0].evidence == "索引可以提高查询效率"
    assert result.metadata["judge_backend"] == "llm"
    assert result.metadata["model"] == MODEL


def test_service_track_is_llm_judge_scorer():
    scorer = _scorer(
        lambda request: httpx.Response(200, json=_ok_response())
    )
    service = SubjectiveScoringService(
        allow_model_load=False,
        llm_judge=_config(),
        scorers={ScoringMode.TEXT: scorer},
    )
    result = service.score(_request())
    assert result.track == "LLMJudgeScorer"
    assert result.score == 5.0


# ---------------------------------------------------------------------------
# 3. 分数越界裁剪、总分封顶、未知 point_id 丢弃、缺失评分点归 0
# ---------------------------------------------------------------------------


def test_score_clipping_total_cap_and_unknown_point_dropped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_response(
                points=[
                    {
                        "point_id": "p1",
                        "score": 99.0,
                        "relation": "supported",
                        "confidence": 2.5,
                    },
                    {
                        "point_id": "p2",
                        "score": -3.0,
                        "relation": "contradicted",
                        "confidence": -1.0,
                    },
                    {"point_id": "ghost", "score": 5.0, "relation": "supported"},
                ]
            ),
        )

    result = _scorer(handler).score(_request())
    # p1 裁剪到 5.0；p2 裁剪到 0.0；ghost 丢弃
    assert result.score == 5.0
    assert result.matched_evidence[0].score == 5.0
    assert result.missed_evidence[0].score == 0.0
    assert any("ghost" in warning for warning in result.warnings)


def test_missing_point_goes_to_missed_with_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        # 只返回 p1，缺失 p2
        return httpx.Response(
            200,
            json=_ok_response(
                points=[
                    {
                        "point_id": "p1",
                        "score": 5.0,
                        "relation": "supported",
                        "confidence": 0.95,
                    }
                ]
            ),
        )

    result = _scorer(handler).score(_request())
    missed_ids = [item.point_id for item in result.missed_evidence]
    assert "p2" in missed_ids
    p2 = next(item for item in result.missed_evidence if item.point_id == "p2")
    assert p2.score == 0.0
    assert p2.relation.value == "unknown"


def test_total_score_capped_at_max_score():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_response(
                points=[
                    {
                        "point_id": "p1",
                        "score": 6.0,
                        "relation": "supported",
                        "confidence": 0.95,
                    },
                    {
                        "point_id": "p2",
                        "score": 6.0,
                        "relation": "supported",
                        "confidence": 0.95,
                    },
                ]
            ),
        )

    result = _scorer(handler).score(_request())
    assert result.score == 10.0
    assert result.max_score == 10.0


# ---------------------------------------------------------------------------
# 4. 非 JSON / 损坏 JSON / HTTP 错误 → 降级或强制人工
# ---------------------------------------------------------------------------


def test_http_error_with_fallback_delegates_to_classic_engine():
    class FakeClassic:
        def score(self, request):
            from subjective_scoring import IntermediateScoreResult

            return IntermediateScoreResult(
                scorer="ClassicStub",
                scoring_mode=request.scoring_mode or ScoringMode.TEXT,
                score=4.0,
                max_score=request.max_score,
                confidence=0.8,
            )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"gateway-secret-body")

    scorer = LLMJudgeScorer(
        config=_config(),
        client=_client(handler),
        fallback=FakeClassic(),
    )
    result = scorer.score(_request())

    assert result.scorer == "ClassicStub"
    assert result.score == 4.0
    assert result.metadata["judge_backend"] == "llm_fallback"
    assert any("LLM judge 失败" in warning for warning in result.warnings)
    assert "gateway-secret-body" not in result.warnings[0]


def test_http_error_without_fallback_forces_manual_review():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"gateway-secret-body")

    scorer = _scorer(handler)
    result = scorer.score(_request())

    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.force_manual_review is True
    assert result.metadata["decision_reason"] == "llm_judge_error"


@pytest.mark.parametrize(
    "content",
    ["not json at all", "no braces here"],
)
def test_invalid_json_forces_manual_review(content):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    scorer = _scorer(handler)
    result = scorer.score(_request())

    assert result.score == 0.0
    assert result.force_manual_review is True


def test_markdown_wrapped_json_is_accepted():
    points_body = {
        "points": [
            {
                "point_id": "p1",
                "score": 5.0,
                "relation": "supported",
                "confidence": 0.95,
            }
        ]
    }
    wrapped = f"```json\n{json.dumps(points_body, ensure_ascii=False)}\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": wrapped}}]},
        )

    result = _scorer(handler).score(_request())
    assert result.score == 5.0


# ---------------------------------------------------------------------------
# 5. judge_backend=reranker → 不发起 HTTP，直接委托经典引擎
# ---------------------------------------------------------------------------


def test_judge_backend_reranker_skips_http():
    class FakeClassic:
        def __init__(self):
            self.calls = 0

        def score(self, request):
            self.calls += 1
            from subjective_scoring import IntermediateScoreResult

            return IntermediateScoreResult(
                scorer="ClassicStub",
                scoring_mode=request.scoring_mode or ScoringMode.TEXT,
                score=6.0,
                max_score=request.max_score,
                confidence=0.9,
            )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("judge_backend=reranker 不应发起 HTTP")

    classic = FakeClassic()
    scorer = LLMJudgeScorer(
        config=_config(),
        client=_client(handler),
        fallback=classic,
    )
    result = scorer.score(
        _request(
            scoring_config={"judge_backend": JudgeBackend.RERANKER.value}
        )
    )

    assert classic.calls == 1
    assert result.scorer == "ClassicStub"
    assert result.score == 6.0


# ---------------------------------------------------------------------------
# 6. 空答案 / 完全一致 → 确定性短路，不发 HTTP
# ---------------------------------------------------------------------------


def test_blank_answer_shortcut_without_http():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("空答案不应发起 HTTP")

    scorer = _scorer(handler)
    result = scorer.score(_request(student_answer=""))

    assert result.score == 0.0
    assert result.confidence == 1.0
    assert result.metadata["decision_reason"] == "blank_answer"


def test_exact_reference_match_shortcut_without_http():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("完全一致答案不应发起 HTTP")

    scorer = _scorer(handler)
    result = scorer.score(
        _request(student_answer="索引可以提高查询效率。")
    )

    assert result.score == 10.0
    assert result.confidence == 1.0
    assert result.metadata["decision_reason"] == "exact_reference_match"


# ---------------------------------------------------------------------------
# 7. 服务级：llm_judge 四题型路由；llm_judge=None 行为不变
# ---------------------------------------------------------------------------


def test_service_routes_all_four_modes_to_llm_judge():
    service = SubjectiveScoringService(
        allow_model_load=False,
        llm_judge=_config(),
        judge_fallback=False,
    )
    for mode, extra in [
        (ScoringMode.TEXT, {"student_answer": "自定义文本作答"}),
        (ScoringMode.SQL, {"student_answer": "SELECT * FROM t", "code_language": "sql"}),
        (
            ScoringMode.CODE,
            {"student_answer": "def f(): pass", "code_language": "python"},
        ),
        (ScoringMode.CALCULATION, {"student_answer": "计算过程"}),
    ]:
        scorer = service._scorers[mode]
        assert isinstance(scorer, LLMJudgeScorer)
        assert scorer.fallback is None


def test_service_without_llm_judge_keeps_classic_routing():
    service = SubjectiveScoringService(allow_model_load=False)
    assert not isinstance(service._scorers[ScoringMode.TEXT], LLMJudgeScorer)
    assert service._scorers[ScoringMode.TEXT].name == "TextRerankerScorer"


# ---------------------------------------------------------------------------
# 8. api_key 不进 repr / 异常文本
# ---------------------------------------------------------------------------


def test_api_key_not_in_client_repr_or_scorer_repr():
    scorer = _scorer(
        lambda request: httpx.Response(200, json=_ok_response())
    )
    assert API_KEY not in repr(scorer.client)
    assert API_KEY not in repr(scorer)
    assert API_KEY not in repr(scorer.config)


def test_api_key_not_in_exception_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"unauthorized")

    with pytest.raises(LLMJudgeRequestError) as caught:
        _client(handler).chat_completions([{"role": "user", "content": "hi"}])
    assert API_KEY not in str(caught.value)
    assert API_KEY not in repr(caught.value)


# ---------------------------------------------------------------------------
# 9. 开放题（scoring_points=[]）：自动单隐式评分点
# ---------------------------------------------------------------------------


def test_open_ended_constructs_implicit_whole_point():
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_ok_response(
                points=[
                    {
                        "point_id": "whole",
                        "score": 8.0,
                        "confidence": 0.9,
                        "reason": "整体作答完整",
                    }
                ]
            ),
        )

    scorer = _scorer(handler)
    result = scorer.score(_request(scoring_points=[]))

    user = requests[0]["messages"][1]["content"]
    assert "whole" in user
    assert result.score == 8.0
    assert result.matched_evidence[0].point_id == "whole"
    assert result.matched_evidence[0].max_score == 10.0
    assert result.confidence == pytest.approx(0.9)
    assert result.metadata["implicit_whole"] is True


def test_open_ended_zero_score_infers_unknown_relation():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_ok_response(
                points=[
                    {
                        "point_id": "whole",
                        "score": 0.0,
                        "relation": "unknown",
                        "confidence": 0.8,
                    }
                ]
            ),
        )

    result = _scorer(handler).score(_request(scoring_points=[]))
    assert result.score == 0.0
    assert result.missed_evidence[0].point_id == "whole"
    assert result.missed_evidence[0].relation.value == "unknown"


# ---------------------------------------------------------------------------
# 10. LLMJudgeClient 传输层：重试 / 超时 / 非 2xx / 非 JSON
# ---------------------------------------------------------------------------


def test_client_retries_429_then_succeeds():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=_ok_response())

    client = LLMJudgeClient(
        config=_config(max_retries=2, retry_backoff_seconds=1.0),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep_fn=sleeps.append,
    )
    body = client.chat_completions(
        [{"role": "user", "content": "hi"}]
    )
    assert calls == 2
    assert sleeps == [2.0]
    assert body["choices"]


def test_client_rejects_invalid_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"student-secret-response")

    client = _client(handler)
    with pytest.raises(LLMJudgeResponseError) as caught:
        client.chat_completions([{"role": "user", "content": "hi"}])
    message = str(caught.value)
    assert API_KEY not in message
    assert "student-secret-response" not in message


def test_client_timeout_is_sanitized():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("student-secret", request=request)

    client = _client(handler)
    with pytest.raises(LLMJudgeRequestError) as caught:
        client.chat_completions([{"role": "user", "content": "hi"}])
    message = str(caught.value)
    assert API_KEY not in message
    assert "student-secret" not in message


def test_client_close_does_not_close_injected_client():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_ok_response())
    )
    raw = httpx.Client(transport=transport)
    client = LLMJudgeClient(config=_config(), client=raw)

    client.close()

    assert raw.is_closed is False
    raw.close()


def test_client_context_manager_closes_owned_client(monkeypatch):
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    client = LLMJudgeClient(config=_config())
    raw = client._client

    with client as active:
        assert active is client
        assert raw.is_closed is False

    assert raw.is_closed is True


# ---------------------------------------------------------------------------
# 11. 服务级 LLM 判分端到端（MockTransport）
# ---------------------------------------------------------------------------


def test_service_end_to_end_llm_judge():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response())

    client = _client(handler)
    service = SubjectiveScoringService(
        allow_model_load=False,
        llm_judge=_config(),
        judge_fallback=True,
    )
    # 用带 MockTransport 的客户端替换默认 LLM 客户端
    for mode in ScoringMode:
        scorer = service._scorers[mode]
        scorer.client = client
        scorer._owns_client = False

    result = service.score(_request())

    assert result.track == "LLMJudgeScorer"
    assert result.score == 5.0
    assert result.decision_reason == "supported_points"
