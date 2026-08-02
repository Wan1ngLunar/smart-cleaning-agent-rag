from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

from utils.checkpoint_handler import create_sqlite_checkpointer


@dataclass
class UnlistedPayload:
    """模拟不在LangGraph安全白名单中的自定义对象。"""

    value: str


def test_sqlite_checkpointer_creates_database_and_tables(
    tmp_path: Path,
):
    """工厂应创建父目录、数据库、关键表和WAL日志模式。"""
    database_path = (
        tmp_path
        / "nested"
        / "agent.sqlite3"
    )
    checkpointer, connection = create_sqlite_checkpointer(
        str(database_path)
    )

    try:
        # 只读查询会触发SqliteSaver自动初始化数据表。
        checkpointer.get_tuple(
            {
                "configurable": {
                    "thread_id": "table-test",
                }
            }
        )

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table'"
            )
        }
        journal_mode = connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]

        assert database_path.is_file()  # sqlite 文件真实生成
        assert "checkpoints" in tables  # 会话断点主表存在
        assert "writes" in tables  # 分步写入记录表存在
        assert journal_mode.lower() == "wal"  # 成功开启WAL并发模式
    finally:
        # Windows测试结束后必须关闭连接，否则临时目录可能无法清理。
        connection.close()


def test_strict_serializer_restores_safe_langchain_message(
    tmp_path: Path,
):
    """安全白名单应允许LangChain消息正常往返。"""
    database_path = tmp_path / "safe.sqlite3"
    checkpointer, connection = create_sqlite_checkpointer(
        str(database_path)
    )

    try:
        message = HumanMessage(
            content="需要跨重启恢复的测试消息"
        )
        serialized = checkpointer.serde.dumps_typed(
            message
        )
        restored = checkpointer.serde.loads_typed(
            serialized
        )

        assert isinstance(restored, HumanMessage)
        assert restored.content == message.content
    finally:
        connection.close()


def test_strict_serializer_does_not_restore_unlisted_type(
    tmp_path: Path,
):
    """白名单外对象可以保存数据，但不能重新实例化原始类型。"""
    database_path = tmp_path / "strict.sqlite3"
    checkpointer, connection = create_sqlite_checkpointer(
        str(database_path)
    )

    try:
        payload = UnlistedPayload(
            value="不应重新实例化"
        )
        serialized = checkpointer.serde.dumps_typed(
            payload
        )
        restored = checkpointer.serde.loads_typed(
            serialized
        )

        # 严格模式只恢复字段数据，不导入并构造UnlistedPayload。
        assert restored == {
            "value": "不应重新实例化"
        }
        assert not isinstance(
            restored,
            UnlistedPayload,
        )
    finally:
        connection.close()
