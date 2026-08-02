import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from agent.react_agent import ReactAgent
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
