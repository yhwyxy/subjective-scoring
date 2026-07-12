from __future__ import annotations

import builtins

import pytest

import subjective_scoring


def test_root_package_imports_without_httpx():
    assert subjective_scoring.CohereRerankerPairScorer is not None


def test_remote_adapter_has_actionable_error_without_httpx(monkeypatch):
    real_import = builtins.__import__

    def import_without_httpx(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_httpx)

    with pytest.raises(ImportError, match=r"subjective-scoring\[remote\]"):
        subjective_scoring.CohereRerankerPairScorer(
            url="https://example.test/v1/rerank",
            api_key="test-key",
            model="test-model",
        )
