"""Cohere-compatible HTTP reranker adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


class RemoteRerankerError(RuntimeError):
    """Base error for remote reranker operations."""


class RemoteRerankerRequestError(RemoteRerankerError):
    """The remote request failed before a usable response was received."""


class RemoteRerankerResponseError(RemoteRerankerError):
    """The remote service returned a malformed response."""


def _load_httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise ImportError(
            "Cloud reranking requires the 'remote' extra: "
            "pip install 'subjective-scoring[remote]'"
        ) from exc
    return httpx


class CohereRerankerPairScorer:
    """Batch pair scorer for Cohere-compatible ``/rerank`` endpoints."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_chunks_per_doc is not None and max_chunks_per_doc <= 0:
            raise ValueError("max_chunks_per_doc must be greater than zero")
        if overlap_tokens is not None and overlap_tokens < 0:
            raise ValueError("overlap_tokens must not be negative")

        self.url = url.strip()
        self.model = model.strip()
        self.name = f"cohere:{self.model}"
        self._api_key = api_key.strip()
        self.max_chunks_per_doc = max_chunks_per_doc
        self.overlap_tokens = overlap_tokens
        self._httpx = _load_httpx()
        self._owns_client = client is None
        self._client = (
            client if client is not None else self._httpx.Client(timeout=timeout)
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(url={self.url!r}, "
            f"model={self.model!r})"
        )

    def __enter__(self) -> CohereRerankerPairScorer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        grouped: dict[str, list[tuple[int, str]]] = {}
        for position, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError("each pair must contain a query and document")
            query, document = pair
            if not isinstance(query, str) or not isinstance(document, str):
                raise ValueError("query and document values must be strings")
            grouped.setdefault(query, []).append((position, document))

        scores = [0.0] * len(pairs)
        for query, entries in grouped.items():
            documents = [document for _, document in entries]
            group_scores = self._score_query(query, documents)
            for (position, _), score in zip(entries, group_scores):
                scores[position] = score
        return scores

    def _score_query(self, query: str, documents: list[str]) -> list[float]:
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        if self.max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = self.max_chunks_per_doc
        if self.overlap_tokens is not None:
            payload["overlap_tokens"] = self.overlap_tokens

        try:
            response = self._client.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except self._httpx.HTTPError:
            raise RemoteRerankerRequestError(
                "remote reranker request failed"
            ) from None

        if response.status_code < 200 or response.status_code >= 300:
            raise RemoteRerankerRequestError(
                f"remote reranker returned HTTP {response.status_code}"
            )

        try:
            body = response.json()
        except ValueError:
            raise RemoteRerankerResponseError(
                "remote reranker returned invalid JSON"
            ) from None

        if not isinstance(body, Mapping):
            raise RemoteRerankerResponseError(
                "remote reranker response must be an object"
            )
        results = body.get("results")
        if not isinstance(results, list):
            raise RemoteRerankerResponseError(
                "remote reranker response results must be a list"
            )

        scores = [0.0] * len(documents)
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, Mapping):
                raise RemoteRerankerResponseError(
                    "remote reranker result must be an object"
                )

            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RemoteRerankerResponseError(
                    "remote reranker result index must be an integer"
                )
            if index < 0 or index >= len(documents):
                raise RemoteRerankerResponseError(
                    "remote reranker result index is out of range"
                )
            if index in seen:
                raise RemoteRerankerResponseError(
                    "remote reranker returned a duplicate result index"
                )

            raw_score = item.get("relevance_score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise RemoteRerankerResponseError(
                    "remote reranker relevance score must be numeric"
                )
            score = float(raw_score)
            if not math.isfinite(score):
                raise RemoteRerankerResponseError(
                    "remote reranker relevance score must be finite"
                )

            seen.add(index)
            scores[index] = float(max(0.0, min(1.0, score)))

        return scores


__all__ = [
    "CohereRerankerPairScorer",
    "RemoteRerankerError",
    "RemoteRerankerRequestError",
    "RemoteRerankerResponseError",
]
