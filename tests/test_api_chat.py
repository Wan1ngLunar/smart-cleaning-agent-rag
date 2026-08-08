from collections.abc import Iterator
from uuid import uuid4

from fastapi.testclient import TestClient

from agent.react_agent import AgentExecutionError
from backend.main import create_app


class FakeChatAgent:
    """模拟Agent的正常、空结果和失败行为。"""

    def __init__(
        self,
        outputs: list[str] | None = None,
        error: AgentExecutionError | None = None,
    ):
        self.outputs = outputs or []
        self.error = error
        self.received_requests: list[
            tuple[str, str]
        ] = []
        self.close_count = 0

    def execute_stream(
        self,
        query: str,
        thread_id: str,
    ) -> Iterator[str]:
        """记录调用参数，并按测试配置产生结果或异常。"""
        self.received_requests.append(
            (
                query,
                thread_id,
            )
        )

        if self.error is not None:
            raise self.error

        yield from self.outputs

    def close(self) -> None:
        """模拟释放Agent资源。"""
        self.close_count += 1


def test_chat_returns_last_non_empty_answer():
    """聊天接口应返回Agent最后一条有效文本。"""
    thread_id = uuid4()
    fake_agent = FakeChatAgent(
        outputs=[
            "中间步骤\n",
            "   ",
            "最终回答\n",
        ],
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "thread_id": str(thread_id),
                "message": "如何清理滤网？",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": str(thread_id),
        "answer": "最终回答",
    }
    assert fake_agent.received_requests == [
        (
            "如何清理滤网？",
            str(thread_id),
        ),
    ]
    assert fake_agent.close_count == 1


def test_chat_returns_safe_agent_error():
    """Agent失败时应返回502、安全说明和问题编号。"""
    request_id = "a1b2c3d4e5f6"
    fake_agent = FakeChatAgent(
        error=AgentExecutionError(
            request_id
        ),
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "thread_id": str(
                    uuid4()
                ),
                "message": "触发测试异常",
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "请求处理暂时失败，请稍后重试。"
            f"问题编号：{request_id}"
        ),
        "request_id": request_id,
    }


def test_chat_returns_safe_empty_response_error():
    """Agent没有文本时应返回统一502，不能伪造答案。"""
    fake_agent = FakeChatAgent(
        outputs=[
            "",
            "   ",
        ],
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "thread_id": str(
                    uuid4()
                ),
                "message": "正常问题",
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Agent没有返回可展示的文本，"
            "请稍后重试。"
        ),
        "request_id": None,
    }


def test_chat_rejects_invalid_request_before_agent_call():
    """非法UUID和空问题不能进入Agent执行阶段。"""
    fake_agent = FakeChatAgent(
        outputs=[
            "不应生成的回答",
        ],
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "thread_id": "not-a-uuid",
                "message": "   ",
            },
        )

    assert response.status_code == 422
    assert fake_agent.received_requests == []
