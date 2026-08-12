import json

import httpx
import pytest
from langchain_core.documents import Document

from rag.reranker import (
    QwenReranker,
    RerankerError,
    build_reranker,
)


def build_documents() -> list[Document]:
    """创建不依赖真实Chroma的候选文档。"""
    return [
        Document(
            id="first",
            page_content="第一份候选资料。",
        ),
        Document(
            id="second",
            page_content="第二份候选资料。",
        ),
    ]


def test_rerank_maps_response_indices_to_documents():
    """接口下标应正确映射回原始LangChain文档。"""

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        request_body = json.loads(
            request.content
        )

        assert request.url.path == (
            "/compatible-api/v1/reranks"
        )
        assert request.headers[
            "authorization"
        ] == "Bearer test-api-key"
        assert request_body["model"] == (
            "qwen3-rerank"
        )
        assert request_body["documents"] == [
            "第一份候选资料。",
            "第二份候选资料。",
        ]
        assert request_body["top_n"] == 2
        assert request_body["instruct"]

        return httpx.Response(
            200,
            json={
                "model": "qwen3-rerank",
                "results": [
                    {
                        "index": 1,
                        "relevance_score": 0.95,
                    },
                    {
                        "index": 0,
                        "relevance_score": 0.40,
                    },
                ],
            },
        )

    transport = httpx.MockTransport(
        handle_request
    )

    with QwenReranker(
        api_key="test-api-key",
        base_url=(
            "https://workspace.example/"
            "compatible-api/v1"
        ),
        transport=transport,
    ) as reranker:
        results = reranker.rerank(
            "哪份资料更相关？",
            build_documents(),
            top_n=2,
        )

    assert [
        result.document.id
        for result in results
    ] == [
        "second",
        "first",
    ]
    assert results[0].original_rank == 2
    assert results[0].rerank_rank == 1
    assert results[0].relevance_score == 0.95


def test_rerank_returns_empty_without_http_request():
    """空候选列表不应调用远程重排序服务。"""

    def reject_request(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "空候选不应发送HTTP请求"
        )

    with QwenReranker(
        api_key="test-api-key",
        base_url="https://workspace.example/v1",
        transport=httpx.MockTransport(
            reject_request
        ),
    ) as reranker:
        assert reranker.rerank(
            "测试问题",
            [],
            top_n=3,
        ) == []


def test_rerank_rejects_empty_query():
    """空问题应在远程调用前被拒绝。"""
    with QwenReranker(
        api_key="test-api-key",
        base_url="https://workspace.example/v1",
    ) as reranker:
        with pytest.raises(
            ValueError,
            match="问题不能为空",
        ):
            reranker.rerank(
                "   ",
                build_documents(),
                top_n=2,
            )


def test_rerank_rejects_invalid_top_n():
    """top_n小于等于0没有合理的排序语义。"""
    with QwenReranker(
        api_key="test-api-key",
        base_url="https://workspace.example/v1",
    ) as reranker:
        with pytest.raises(
            ValueError,
            match="top_n必须是大于0的整数",
        ):
            reranker.rerank(
                "测试问题",
                build_documents(),
                top_n=0,
            )


def test_rerank_wraps_http_error_without_api_key():
    """HTTP错误应保留状态和问题编号，但不能暴露密钥。"""

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "request_id": "request-123",
                "message": "内部错误",
            },
        )

    with QwenReranker(
        api_key="secret-api-key",
        base_url="https://workspace.example/v1",
        transport=httpx.MockTransport(
            handle_request
        ),
    ) as reranker:
        with pytest.raises(
            RerankerError,
            match="HTTP 500.*request-123",
        ) as error_info:
            reranker.rerank(
                "测试问题",
                build_documents(),
                top_n=2,
            )

    assert "secret-api-key" not in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "raw_results",
    [
        [
            {
                "index": 5,
                "relevance_score": 0.90,
            }
        ],
        [
            {
                "index": 0,
                "relevance_score": 0.90,
            },
            {
                "index": 0,
                "relevance_score": 0.80,
            },
        ],
        [
            {
                "index": 0,
                "relevance_score": 1.50,
            }
        ],
    ],
)
def test_rerank_rejects_invalid_response_results(
    raw_results: list[dict],
):
    """越界、重复下标和非法分数都不能进入RAG上下文。"""

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": raw_results,
            },
        )

    with QwenReranker(
        api_key="test-api-key",
        base_url="https://workspace.example/v1",
        transport=httpx.MockTransport(
            handle_request
        ),
    ) as reranker:
        with pytest.raises(RerankerError):
            reranker.rerank(
                "测试问题",
                build_documents(),
                top_n=2,
            )

def test_build_reranker_uses_environment_and_yaml_config():
    """工厂应组合内存环境变量和公开YAML配置。"""

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        request_body = json.loads(
            request.content
        )

        assert request.url.path == (
            "/compatible-api/v1/reranks"
        )
        assert request.headers[
            "authorization"
        ] == "Bearer factory-test-key"
        assert request_body["model"] == (
            "qwen3-rerank"
        )
        assert request_body["instruct"]

        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.90,
                    }
                ]
            },
        )

    environment = {
        "DASHSCOPE_API_KEY": (
            "factory-test-key"
        ),
        "DASHSCOPE_RERANK_BASE_URL": (
            "https://workspace.example/"
            "compatible-api/v1"
        ),
    }

    with build_reranker(
        environment=environment,
        transport=httpx.MockTransport(
            handle_request
        ),
    ) as reranker:
        results = reranker.rerank(
            "测试问题",
            build_documents(),
            top_n=1,
        )

    assert len(results) == 1
    assert results[0].document.id == "first"


@pytest.mark.parametrize(
    (
        "environment",
        "expected_message",
    ),
    [
        (
            {
                "DASHSCOPE_RERANK_BASE_URL": (
                    "https://workspace.example/"
                    "compatible-api/v1"
                ),
            },
            "缺少DASHSCOPE_API_KEY",
        ),
        (
            {
                "DASHSCOPE_API_KEY": (
                    "test-api-key"
                ),
            },
            "缺少DASHSCOPE_RERANK_BASE_URL",
        ),
    ],
)
def test_build_reranker_rejects_missing_environment(
    environment: dict[str, str],
    expected_message: str,
):
    """缺少密钥或接口地址时应在创建连接池前失败。"""
    with pytest.raises(
        RuntimeError,
        match=expected_message,
    ):
        build_reranker(
            environment=environment
        )
