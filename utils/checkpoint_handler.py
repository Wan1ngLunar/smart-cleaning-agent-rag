"""创建并管理本地SQLite会话检查点连接。"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from utils.path_tool import get_abs_path


def create_sqlite_checkpointer(
    database_path: str,
) -> tuple[SqliteSaver, sqlite3.Connection]:
    """创建安全的SQLite Checkpointer，并返回需要显式关闭的连接。"""
    # 配置路径转换为项目绝对路径，避免从不同目录启动时生成多份数据库。
    absolute_path = Path(
        get_abs_path(database_path)
    )

    # SQLite只能创建数据库文件，不能自动创建缺失的父目录。
    absolute_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(absolute_path),
        # Streamlit和LangGraph可能从不同线程访问同一个连接。默认 sqlite 限制单线程访问连接；项目里 Streamlit 界面、LangGraph 运行在不同线程，关闭线程检测，允许跨线程共用连接。
        check_same_thread=False,
        # 数据库短暂被占用时最多等待30秒，减少并发写入立即失败。
        timeout=30,
    )

    try:
        # WAL允许读取和写入更好地并行，适合本地多会话演示。
        connection.execute(
            "PRAGMA journal_mode=WAL"
        )

        # None表示只允许LangGraph内置安全白名单中的类型反序列化。
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=None
        )
        checkpointer = SqliteSaver(
            connection,
            serde=serializer,
        )

        # 调用方同时持有连接，应用结束时必须主动关闭它。
        return checkpointer, connection
    except Exception:
        # 初始化中途失败时立即释放文件句柄，避免数据库被长期占用。
        connection.close()
        raise
