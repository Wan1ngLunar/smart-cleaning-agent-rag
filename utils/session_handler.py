"""创建和校验用于持久化会话的thread_id。"""

from uuid import UUID, uuid4


def create_thread_id() -> str:
    """创建标准UUID字符串作为新的会话标识。"""
    return str(uuid4())


def normalize_thread_id(
    raw_thread_id: str | None,
) -> str:
    """保留合法UUID；缺失或非法时创建新的会话标识。"""
    if not raw_thread_id:
        return create_thread_id()

    # 标准UUID字符串长度为36，提前拒绝异常长的URL参数。
    if len(raw_thread_id) > 36:
        return create_thread_id()

    try:
        # 解析后重新转成标准小写、带连字符格式。
        return str(UUID(raw_thread_id))
    except ValueError:
        # URL被手工修改或参数损坏时创建隔离的新会话。
        return create_thread_id()
