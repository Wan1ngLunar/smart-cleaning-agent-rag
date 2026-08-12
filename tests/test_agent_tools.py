import json
import logging
from datetime import date

import pytest

import agent.tools.agent_tools as tools_module
from agent.tools.agent_tools import (
    external_data,
    fetch_external_data,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    load_external_data,
)

# 项目源码统一使用 UTF-8，因此测试常量直接使用可读中文。
SHENZHEN = "深圳"
DEMO = "演示"
NOT_REAL_TIME = "非实时"


@pytest.fixture(autouse=True)
def reset_external_data_cache():
    """确保每个测试都从空缓存开始，避免测试顺序影响结果。"""
    external_data.clear()

    yield

    # 测试结束后再次清理，避免状态泄漏到其他测试模块。
    external_data.clear()


def test_demo_context_tools_are_deterministic():
    """演示身份、城市、月份和天气来源应稳定且透明。"""
    assert get_user_id.invoke({}) == "1001"
    assert get_user_location.invoke({}) == SHENZHEN
    assert get_current_month.invoke({}) == date.today().strftime("%Y-%m")

    weather = get_weather.invoke({"city": SHENZHEN})

    assert DEMO in weather
    assert NOT_REAL_TIME in weather
    assert DEMO in get_weather.description


def test_external_data_load_is_idempotent():
    """重复加载同一 CSV 不应增加重复记录。"""
    load_external_data()
    first_count = sum(
        len(month_records)
        for month_records in external_data.values()
    )

    load_external_data()
    second_count = sum(
        len(month_records)
        for month_records in external_data.values()
    )

    assert len(external_data) == 10
    assert first_count == 120
    assert second_count == first_count


def test_fetch_external_data_returns_json_string():
    """工具应清理参数空格，并按声明返回 JSON 字符串。"""
    result = fetch_external_data.invoke({
        "user_id": " 1001 ",  # 故意前后带空格，验证工具的参数归一化。
        "month": " 2025-06 ",
    })

    assert isinstance(result, str)

    record = json.loads(result)

    assert set(record) == {
        "特征",
        "效率",
        "耗材",
        "对比",
    }


def test_fetch_external_data_returns_empty_string_when_missing(
    caplog: pytest.LogCaptureFixture,
):
    """未命中时应返回空字符串，且日志不能保存查询参数。"""
    private_user_id = "private-user-1001"
    private_month = "2099-01"

    with caplog.at_level(
        logging.WARNING,
        logger="agent",
    ):
        result = fetch_external_data.invoke({
            "user_id": private_user_id,
            "month": private_month,
        })

    assert result == ""
    assert "未检索到匹配的演示使用记录" in (
        caplog.text
    )

    # 工具参数可能来自用户输入，不能写入持久化日志。
    assert private_user_id not in caplog.text
    assert private_month not in caplog.text

def test_rag_service_is_lazily_created_reused_and_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    """RAG服务应首次调用时创建，后续复用，并能显式关闭。"""

    class StubRagService:
        """替代真实RAG服务，避免测试访问Chroma和外部接口。"""

        def __init__(self):
            self.received_queries: list[str] = []
            self.close_count = 0

        def rag_summarize(
            self,
            query: str,
        ) -> str:
            """记录工具传入的问题并返回固定测试回答。"""
            self.received_queries.append(query)
            return f"测试回答：{query}"

        def close(self) -> None:
            """记录资源关闭次数。"""
            self.close_count += 1

    created_services: list[StubRagService] = []

    def create_stub_rag_service() -> StubRagService:
        """记录延迟初始化实际创建的服务实例。"""
        service = StubRagService()
        created_services.append(service)
        return service

    # 确保测试开始前没有其他测试遗留的共享服务。
    tools_module.close_rag_service()

    # 替换构造函数，保证测试不会连接真实知识库或重排序接口。
    monkeypatch.setattr(
        tools_module,
        "RagSummarizeService",
        create_stub_rag_service,
    )

    # 只修改构造函数不会创建服务，证明初始化确实是延迟的。
    assert created_services == []

    first_service = tools_module.get_rag_service()
    second_service = tools_module.get_rag_service()

    # 连续获取两次仍然只有一个实例，证明服务被进程内复用。
    assert first_service is second_service
    assert len(created_services) == 1

    result = tools_module.rag_summarize.invoke(
        {
            "query": "如何清理滤网？",
        }
    )

    assert result == "测试回答：如何清理滤网？"
    assert first_service.received_queries == [
        "如何清理滤网？"
    ]

    tools_module.close_rag_service()
    tools_module.close_rag_service()

    # 重复关闭不会重复释放同一个服务。
    assert first_service.close_count == 1
    assert tools_module._rag_service is None
