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


@pytest.fixture(autouse=True)
def reset_external_data_cache():
    external_data.clear()

    yield

    external_data.clear()


def test_demo_context_tools_are_deterministic():
    assert get_user_id.invoke({}) == "1001"
    assert get_user_location.invoke({}) == "\u6df1\u5733"
    assert get_current_month.invoke({}) == date.today().strftime("%Y-%m")

    weather = get_weather.invoke({"city": "\u6df1\u5733"})

    assert "\u6f14\u793a" in weather
    assert "\u975e\u5b9e\u65f6" in weather
    assert "\u6f14\u793a" in get_weather.description


def test_external_data_load_is_idempotent():
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
    result = fetch_external_data.invoke({
        "user_id": " 1001 ",  #  故意前后带空格
        "month": " 2025-06 ",
    })

    assert isinstance(result, str)

    record = json.loads(result)

    assert set(record) == {
        "\u7279\u5f81",
        "\u6548\u7387",
        "\u8017\u6750",
        "\u5bf9\u6bd4",
    }


def test_fetch_external_data_returns_empty_string_when_missing():
    result = fetch_external_data.invoke({
        "user_id": "1001",
        "month": "2099-01",#  不存在的月份
    })

    assert result == ""