import json

import httpx
import pytest

from frontend.api_client import (
    ApiClientError,
    BackendApiClient,
)


def test_api_client_reads_history():
    """客户端应读取并整理后端公开历史消息。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"
        assert (
            request.url.path
            == (
                "/api/v1/sessions/"
                "test-thread/messages"
            )
        )

        return httpx.Response(
            200,
            json={
                "thread_id": "test-thread",
                "messages": [
                    {
                        "role": "user",
                        "content": "  如何清理滤网？  ",
                    },
                    {
                        "role": "assistant",
                        "content": "请定期清理。",
                    },
                ],
            },
        )

    client = BackendApiClient(
        base_url="http://backend.test",
        transport=httpx.MockTransport(
            handler
        ),
    )

    try:
        messages = client.get_history(
            "test-thread"
        )
    finally:
        client.close()

    assert messages == [
        {
            "role": "user",
            "content": "如何清理滤网？",
        },
        {
            "role": "assistant",
            "content": "请定期清理。",
        },
    ]


def test_api_client_sends_chat_request():
    """客户端应发送正确JSON并返回最终回答。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/chat"

        request_payload = json.loads(
            request.content.decode("utf-8")
        )

        assert request_payload == {
            "thread_id": "test-thread",
            "message": "如何保养滤网？",
        }

        return httpx.Response(
            200,
            json={
                "thread_id": "test-thread",
                "answer": "建议每周清理滤网。",
            },
        )

    client = BackendApiClient(
        base_url="http://backend.test/",
        transport=httpx.MockTransport(
            handler
        ),
    )

    try:
        answer = client.chat(
            thread_id="test-thread",
            message="如何保养滤网？",
        )
    finally:
        client.close()

    assert answer == "建议每周清理滤网。"


def test_api_client_preserves_safe_backend_error():
    """后端安全错误和问题编号应传递给页面。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            502,
            json={
                "detail": (
                    "请求处理暂时失败，请稍后重试。"
                    "问题编号：a1b2c3d4e5f6"
                ),
                "request_id": "a1b2c3d4e5f6",
            },
        )

    client = BackendApiClient(
        base_url="http://backend.test",
        transport=httpx.MockTransport(
            handler
        ),
    )

    try:
        with pytest.raises(
            ApiClientError,
            match="a1b2c3d4e5f6",
        ):
            client.chat(
                "test-thread",
                "测试问题",
            )
    finally:
        client.close()


def test_api_client_wraps_connection_error():
    """无法连接后端时不能把HTTPX异常直接暴露给页面。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "测试连接失败",
            request=request,
        )

    client = BackendApiClient(
        base_url="http://backend.test",
        transport=httpx.MockTransport(
            handler
        ),
    )

    try:
        with pytest.raises(
            ApiClientError,
            match="无法连接后端服务",
        ) as error_info:
            client.get_history(
                "test-thread"
            )
    finally:
        client.close()

    # 原始HTTPX错误仍保留在异常因果链中，方便开发排查。
    assert isinstance(
        error_info.value.__cause__,
        httpx.ConnectError,
    )


def test_api_client_rejects_invalid_payload():
    """成功状态下的错误响应结构也必须被拒绝。"""

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "unexpected": "错误结构",
            },
        )

    client = BackendApiClient(
        base_url="http://backend.test",
        transport=httpx.MockTransport(
            handler
        ),
    )

    try:
        with pytest.raises(
            ApiClientError,
            match="缺少消息列表",
        ):
            client.get_history(
                "test-thread"
            )
    finally:
        client.close()
