import math
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts


@dataclass(frozen=True)
class ToolRoutingCase:
    """保存一条首步工具路由评测用例。"""

    case_id: str
    query: str

    # 空元组表示模型不应该调用任何工具。
    expected_tools: tuple[str, ...]


# 评估模型看到用户第一条问题时，能否选择正确的起始工具。
TOOL_ROUTING_CASES = (
    ToolRoutingCase(
        case_id="rag_maintenance",
        query="扫地机器人的HEPA滤网应该多久更换一次？",
        expected_tools=("rag_summarize",),
    ),
    ToolRoutingCase(
        case_id="rag_troubleshooting",
        query="设置禁区后机器人仍然进入，应该怎样排查？",
        expected_tools=("rag_summarize",),
    ),
    ToolRoutingCase(
        case_id="explicit_demo_weather",
        query="请查询深圳的演示天气，并明确说明是否实时。",
        expected_tools=("get_weather",),
    ),
    ToolRoutingCase(
        case_id="weather_without_city",
        query="当前演示用户所在城市的天气怎么样？",
        expected_tools=("get_user_location",),
    ),
    ToolRoutingCase(
        case_id="demo_location",
        query="当前演示用户所在的城市是什么？",
        expected_tools=("get_user_location",),
    ),
    ToolRoutingCase(
        case_id="demo_user_id",
        query="当前演示用户的ID是什么？",
        expected_tools=("get_user_id",),
    ),
    ToolRoutingCase(
        case_id="current_month",
        query="系统当前月份是什么？请使用YYYY-MM格式。",
        expected_tools=("get_current_month",),
    ),
    ToolRoutingCase(
        case_id="specified_month_report",
        query="请生成我的2025-06使用报告。",
        # 报告固定流程的第一步是获取演示用户ID。
        expected_tools=("get_user_id",),
    ),
    ToolRoutingCase(
        case_id="current_month_report",
        query="请生成我的当前月份使用报告。",
        # 即使月份未知，报告流程仍要求先获取用户ID。
        expected_tools=("get_user_id",),
    ),
    ToolRoutingCase(
        case_id="ordinary_greeting",
        query="你好，请简单介绍一下你能做什么。",
        expected_tools=(),
    ),
    ToolRoutingCase(
        case_id="unrelated_cooking",
        query="番茄炒蛋应该放多少糖？",
        expected_tools=(),
    ),
    ToolRoutingCase(
        case_id="thanks_without_tool",
        query="谢谢你的帮助。",
        expected_tools=(),
    ),
)


AVAILABLE_TOOLS = (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)


def extract_tool_names(
    message: AIMessage,
) -> tuple[str, ...]:
    """按模型返回顺序提取首轮工具调用名称。"""
    return tuple(
        str(tool_call["name"])
        for tool_call in message.tool_calls
    )


def extract_token_usage(
    message: AIMessage,
) -> tuple[int, int, int]:
    """读取一次首步路由模型调用的输入、输出和总Token数。"""
    usage = message.usage_metadata or {} # 大模型返回的 token 消耗统计。

    input_tokens = int(
        usage.get("input_tokens", 0)
    )
    output_tokens = int(
        usage.get("output_tokens", 0)
    )
    total_tokens = int(
        usage.get(
            "total_tokens",
            input_tokens + output_tokens,
        )
    )

    return (
        input_tokens,
        output_tokens,
        total_tokens,
    )


def calculate_nearest_rank_percentile(
    values: list[int],
    percentile: float,
) -> int:
    """使用最近排名法计算Token分布百分位。"""
    if not values:
        raise ValueError(
            "计算百分位数时至少需要一个数值"
        )

    if not 0 < percentile <= 1:
        raise ValueError(
            "percentile必须在0到1之间"
        )

    sorted_values = sorted(values)
    rank = math.ceil(
        len(sorted_values) * percentile
    )

    return sorted_values[rank - 1]


if __name__ == "__main__":
    # 只让模型选择工具，不执行工具，因此不会访问RAG或CSV。
    router_model = chat_model.bind_tools(
        list(AVAILABLE_TOOLS)
    ) # LangChain 语法，把工具 schema 绑定到大模型，告诉模型可用工具。
    system_prompt = load_system_prompts()

    passed_count = 0
    # 三个列表，保存每一轮的 token，后面用来算平均、P95。
    input_token_counts: list[int] = []
    output_token_counts: list[int] = []
    total_token_counts: list[int] = []

    for case in TOOL_ROUTING_CASES:
        response = router_model.invoke(
            [
                SystemMessage(
                    content=system_prompt,
                ),
                HumanMessage(
                    content=case.query,
                ),
            ]
        )

        if not isinstance(response, AIMessage):
            raise TypeError(
                f"{case.case_id}没有返回AIMessage"
            )

        actual_tools = extract_tool_names(
            response
        )
        passed = (
            actual_tools
            == case.expected_tools
        )

        if passed:
            passed_count += 1

        (
            input_tokens,
            output_tokens,
            total_tokens,
        ) = extract_token_usage(response)

        input_token_counts.append(
            input_tokens
        )
        output_token_counts.append(
            output_tokens
        )
        total_token_counts.append(
            total_tokens
        )

        print("=" * 80)
        print("用例：", case.case_id)
        print("问题：", case.query)
        print(
            "预期工具：",
            ", ".join(case.expected_tools)
            if case.expected_tools
            else "不调用工具",
        )
        print(
            "实际工具：",
            ", ".join(actual_tools)
            if actual_tools
            else "未调用工具",
        )
        print(
            "评估结果：",
            "通过" if passed else "失败",
        )
        print("输入Token：", input_tokens)
        print("输出Token：", output_tokens)
        print("总Token：", total_tokens)

    case_count = len(
        TOOL_ROUTING_CASES
    )
    routing_accuracy = (
        passed_count / case_count
    ) # 路由准确率 = 通过数 / 全部用例数
    average_total_tokens = (
        sum(total_token_counts)
        / case_count
    ) # 平均每次路由消耗 token
    p95_total_tokens = (
        calculate_nearest_rank_percentile(
            total_token_counts,
            0.95,
        )
    )

    print()
    print("首步工具路由评估")
    print(
        "工具路由成功率："
        f"{passed_count}/{case_count}，"
        f"{routing_accuracy:.2%}"
    )
    print(
        "平均单次首步路由Token："
        f"{average_total_tokens:.2f}"
    )
    print(
        "P95单次首步路由Token：",
        p95_total_tokens,
    )
    print(
        "平均输入Token："
        f"{sum(input_token_counts) / case_count:.2f}"
    )
    print(
        "平均输出Token："
        f"{sum(output_token_counts) / case_count:.2f}"
    )

    if passed_count != case_count:
        raise SystemExit(1)
