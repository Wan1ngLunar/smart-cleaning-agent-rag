from uuid import UUID

from utils.session_handler import (
    create_thread_id,
    normalize_thread_id,
)


def test_normalize_thread_id_keeps_valid_uuid():
    """合法UUID应保持同一个会话，不应被替换。"""
    thread_id = create_thread_id()

    assert normalize_thread_id(thread_id) == thread_id


def test_normalize_thread_id_replaces_missing_and_invalid_values():
    """缺失、非法或异常长的参数应替换为新的合法UUID。"""
    generated_ids = (
        normalize_thread_id(None),
        normalize_thread_id("不是UUID"),
        normalize_thread_id("x" * 100),
    )

    for thread_id in generated_ids:
        # UUID构造成功说明返回值格式合法。
        assert str(UUID(thread_id)) == thread_id

    # 每个非法输入都应创建独立会话，不能共享固定默认ID。
    assert len(set(generated_ids)) == len(
        generated_ids
    )
