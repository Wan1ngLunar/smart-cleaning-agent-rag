from uuid import uuid4

from fastapi.testclient import TestClient

from backend.main import create_app


class FakeHistoryAgent:
    """模拟带历史读取能力的Agent。"""

    def __init__(
        self,
        messages: list[dict[str, str]],
    ):
        self.messages = messages
        self.received_thread_ids: list[str] = []
        self.close_count = 0

    def get_history(
        self,
        thread_id: str,
    ) -> list[dict[str, str]]:
        """记录会话ID并返回预设公开消息。"""
        self.received_thread_ids.append(
            thread_id
        )

        # 返回副本，避免接口代码意外修改测试预设数据。
        return [
            message.copy()
            for message in self.messages
        ]

    def close(self) -> None:
        """模拟释放后端Agent资源。"""
        self.close_count += 1


def test_session_history_returns_public_messages():
    """合法会话应返回用户和助手的公开历史。"""
    thread_id = uuid4()
    fake_agent = FakeHistoryAgent(
        messages=[
            {
                "role": "user",
                "content": "如何保养滤网？",
            },
            {
                "role": "assistant",
                "content": "建议定期清理滤网。",
            },
        ],
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/sessions/{thread_id}/messages"
        )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": str(thread_id),
        "messages": [
            {
                "role": "user",
                "content": "如何保养滤网？",
            },
            {
                "role": "assistant",
                "content": "建议定期清理滤网。",
            },
        ],
    }
    assert fake_agent.received_thread_ids == [
        str(thread_id),
    ]
    assert fake_agent.close_count == 1


def test_session_history_returns_empty_list():
    """新会话没有历史时应返回空列表，而不是404。"""
    thread_id = uuid4()
    fake_agent = FakeHistoryAgent(
        messages=[],
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/sessions/{thread_id}/messages"
        )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": str(thread_id),
        "messages": [],
    }


def test_session_history_rejects_invalid_uuid():
    """非法会话ID应在调用Agent之前返回422。"""
    fake_agent = FakeHistoryAgent(
        messages=[],
    )
    application = create_app(
        agent_factory=lambda: fake_agent,
    )

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/sessions/not-a-uuid/messages"
        )

    assert response.status_code == 422

    # 参数校验失败后，不能读取任何SQLite会话。
    assert fake_agent.received_thread_ids == []
