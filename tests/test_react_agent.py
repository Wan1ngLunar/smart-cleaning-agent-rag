import logging
import re
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from agent.react_agent import (
    AgentExecutionError,
    ReactAgent,
)
from utils.config_handler import agent_conf


def test_react_agent_restores_thread_across_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """关闭并重建Agent后，相同thread_id应恢复已保存消息。"""
    database_path = tmp_path / "agent.sqlite3"

    # 测试使用临时数据库，不接触正式storage中的会话数据。
    monkeypatch.setitem(
        agent_conf,
        "checkpoint_path",
        str(database_path),
    )

    thread_config = {
        "configurable": {
            "thread_id": "persistent-thread",
        }
    }
    other_thread_config = {
        "configurable": {
            "thread_id": "other-thread",
        }
    }

    first_agent = ReactAgent()

    try:
        # 直接更新图状态，不调用真实聊天模型。
        first_agent.agent.update_state(
            thread_config,
            {
                "messages": [
                    HumanMessage(
                        content="我家里养了两只猫"
                    )
                ]
            },
        )
    finally:
        first_agent.close()

    second_agent = ReactAgent()

    try:
        restored_state = second_agent.agent.get_state(
            thread_config
        )
        restored_messages = restored_state.values.get(
            "messages",
            [],
        )
        human_contents = [
            message.content
            for message in restored_messages
            if isinstance(message, HumanMessage)
        ]

        assert "我家里养了两只猫" in human_contents

        # 不同thread_id不能读取另一个会话的消息。
        other_state = second_agent.agent.get_state(
            other_thread_config
        )
        assert not other_state.values.get(
            "messages",
            [],
        )
    finally:
        second_agent.close()


def test_react_agent_close_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """连接可以重复关闭，且第一次关闭后文件句柄确实失效。"""
    database_path = tmp_path / "close.sqlite3"
    monkeypatch.setitem(
        agent_conf,
        "checkpoint_path",
        str(database_path),
    )

    agent = ReactAgent()
    connection = agent._checkpoint_connection

    agent.close()
    agent.close()

    # 保存关闭前的连接引用，确认底层SQLite连接已经不可使用。
    with pytest.raises(
        sqlite3.ProgrammingError,
        match="closed",
    ):
        connection.execute("SELECT 1")

def test_react_agent_history_only_returns_visible_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """页面历史应过滤工具调用和工具返回等内部消息。"""
    database_path = tmp_path / "history.sqlite3"
    monkeypatch.setitem(
        agent_conf,
        "checkpoint_path",
        str(database_path),
    )

    agent = ReactAgent()
    thread_config = {
        "configurable": {
            "thread_id": "history-thread",
        }
    }

    try:
        agent.agent.update_state(
            thread_config,
            {
                "messages": [
                    HumanMessage(
                        content="深圳天气怎么样？"
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_weather",
                                "args": {
                                    "city": "深圳",
                                },
                                "id": "weather-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="演示天气为晴天",
                        tool_call_id="weather-call",
                    ),
                    AIMessage(
                        content="深圳演示天气为晴天。"
                    ),
                ]
            },
        )

        history = agent.get_history(
            "history-thread"
        )

        assert history == [
            {
                "role": "user",
                "content": "深圳天气怎么样？",
            },
            {
                "role": "assistant",
                "content": "深圳演示天气为晴天。",
            },
        ]
    finally:
        agent.close()

def test_execute_stream_logs_success_without_query(
    caplog: pytest.LogCaptureFixture,
):
    """成功请求应记录编号和耗时，但不能记录用户完整问题。"""
    observed_context: dict[str, object] = {}

    class SuccessfulAgent:
        """提供固定流式结果，避免测试调用真实模型。"""

        def stream(self, *args, **kwargs):
            # 保存运行时上下文，用于验证请求编号是否传给中间件。
            observed_context.update(
                kwargs["context"]
            )

            yield {
                "messages": [
                    AIMessage(
                        content="这是测试回答"
                    )
                ]
            }

    # 绕过正式初始化，避免创建SQLite连接和真实LangGraph Agent。
    react_agent = ReactAgent.__new__(
        ReactAgent
    )
    react_agent.agent = SuccessfulAgent()

    private_query = "这是一条不应进入日志的用户问题"

    with caplog.at_level(
        logging.INFO,
        logger="agent",
    ):
        result = list(
            react_agent.execute_stream(
                private_query,
                thread_id="12345678-secret-session",
            )
        )

    assert result == [
        "这是测试回答\n"
    ]
    assert re.fullmatch(
        r"[0-9a-f]{12}",
        str(observed_context["request_id"]),
    )
    assert observed_context["session_id"] == "12345678"
    assert "请求开始" in caplog.text
    assert "请求成功" in caplog.text
    assert "elapsed_ms=" in caplog.text

    # 日志可以记录编号和会话前缀，但不能记录完整用户问题。
    assert private_query not in caplog.text
    assert "secret-session" not in caplog.text

def test_execute_stream_wraps_failure_with_request_id(
    caplog: pytest.LogCaptureFixture,
):
    """底层异常应写入日志，但页面只能收到安全提示。"""

    class FailingAgent:
        """模拟模型连接失败，不发起任何真实网络请求。"""

        def stream(self, *args, **kwargs):
            raise ConnectionError(
                "模拟的上游连接失败"
            )

    # 绕过正式初始化，让测试只关注execute_stream异常边界。
    react_agent = ReactAgent.__new__(
        ReactAgent
    )
    react_agent.agent = FailingAgent()

    with caplog.at_level(
        logging.ERROR,
        logger="agent",
    ):
        with pytest.raises(
            AgentExecutionError
        ) as captured_error:
            list(
                react_agent.execute_stream(
                    "测试问题",
                    thread_id="failure-session",
                )
            )

    safe_error = captured_error.value

    assert re.fullmatch(
        r"[0-9a-f]{12}",
        safe_error.request_id,
    )
    assert safe_error.request_id in (
        safe_error.public_message
    )
    assert "模拟的上游连接失败" not in (
        safe_error.public_message
    )

    # 服务端日志保留编号和异常类型，开发者可以据此排查。
    assert safe_error.request_id in caplog.text
    assert "error_type=ConnectionError" in (
        caplog.text
    )

    # 保留异常因果链，但页面不会直接展示该底层异常。
    assert isinstance(
        safe_error.__cause__,
        ConnectionError,
    )
