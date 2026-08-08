from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.schemas import (
    ChatRequest,
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
)


def test_chat_request_accepts_valid_uuid_and_strips_message():
    """合法请求应保留会话ID，并清理问题两侧的空白。"""
    thread_id = uuid4()

    request = ChatRequest(
        thread_id=str(thread_id),
        message="  扫地机器人应该怎样保养？  ",
    )

    assert request.thread_id == thread_id
    assert request.message == "扫地机器人应该怎样保养？"


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
        "问" * 4001,
    ],
)
def test_chat_request_rejects_invalid_message(
    message,
):
    """空问题、纯空格和超过限制的问题都应被拒绝。"""
    with pytest.raises(ValidationError):
        ChatRequest(
            thread_id=str(uuid4()),
            message=message,
        )


def test_chat_request_rejects_unknown_field():
    """接口契约之外的字段不能被静默忽略。"""
    with pytest.raises(ValidationError):
        ChatRequest(
            thread_id=str(uuid4()),
            message="正常问题",
            unexpected_field="不允许的字段",
        )


def test_history_response_serializes_public_messages():
    """历史响应只应包含会话ID和可展示的角色、正文。"""
    thread_id = uuid4()

    response = HistoryResponse(
        thread_id=thread_id,
        messages=[
            {
                "role": "user",
                "content": "如何清理滤网？",
            },
            {
                "role": "assistant",
                "content": "请先关闭设备电源。",
            },
        ],
    )

    payload = response.model_dump(
        mode="json",
    )

    assert payload == {
        "thread_id": str(thread_id),
        "messages": [
            {
                "role": "user",
                "content": "如何清理滤网？",
            },
            {
                "role": "assistant",
                "content": "请先关闭设备电源。",
            },
        ],
    }


def test_response_models_reject_invalid_fixed_values():
    """健康状态和问题编号必须符合公开接口约定。"""
    with pytest.raises(ValidationError):
        HealthResponse(
            status="unknown",
        )

    with pytest.raises(ValidationError):
        ErrorResponse(
            detail="请求失败",
            request_id="错误编号",
        )
