import json
from datetime import date

import pytest

from agent.tools.agent_tools import (
    external_data,
    fetch_external_data,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    load_external_data,
)

# Unicode 转义可避免 Windows 终端编码影响测试源码中的中文断言。
SHENZHEN = "\u6df1\u5733"
DEMO = "\u6f14\u793a"
NOT_REAL_TIME = "\u975e\u5b9e\u65f6"


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

    # 分别是“特征、效率、耗材、对比”四个字段。
    assert set(record) == {
        "\u7279\u5f81",
        "\u6548\u7387",
        "\u8017\u6750",
        "\u5bf9\u6bd4",
    }


def test_fetch_external_data_returns_empty_string_when_missing():
    """不存在的月份必须返回空字符串，不能编造或随机改查其他月份。"""
    result = fetch_external_data.invoke({
        "user_id": "1001",
        "month": "2099-01",
    })

    assert result == ""
