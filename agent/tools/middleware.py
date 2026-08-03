# Callable用于标注中间件接收的处理函数。
from collections.abc import Callable
from time import perf_counter

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import (
    load_report_prompts,
    load_system_prompts,
)


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage | Command,
    ],
) -> ToolMessage | Command:
    """记录工具名称、请求编号、结果和耗时，不记录工具参数。"""
    tool_name = request.tool_call["name"]
    request_id = str(
        request.runtime.context.get(
            "request_id",
            "unknown",
        )
    )
    started_at = perf_counter()

    logger.info(
        "[tool_call]调用开始 "
        "request_id=%s tool=%s",
        request_id,
        tool_name,
    )

    try:
        result = handler(request)
    except Exception as error:
        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000

        # 记录异常堆栈和类型，但不主动记录工具入参。
        logger.exception(
            "[tool_call]调用失败 "
            "request_id=%s tool=%s "
            "elapsed_ms=%.2f error_type=%s",
            request_id,
            tool_name,
            elapsed_ms,
            type(error).__name__,
        )

        # 使用原始raise保留最准确的异常调用栈。
        raise

    elapsed_ms = (
        perf_counter() - started_at
    ) * 1000

    logger.info(
        "[tool_call]调用成功 "
        "request_id=%s tool=%s "
        "elapsed_ms=%.2f",
        request_id,
        tool_name,
        elapsed_ms,
    )

    if tool_name == "fill_context_for_report":
        # 报告工具成功后，为同一次请求切换报告专用Prompt。
        request.runtime.context["report"] = True

    return result


@wrap_model_call
def monitor_model(
    request: ModelRequest,
    handler: Callable[
        [ModelRequest],
        ModelResponse,
    ],
) -> ModelResponse:
    """记录模型调用次数、请求编号、结果和耗时。"""
    request_id = str(
        request.runtime.context.get(
            "request_id",
            "unknown",
        )
    )
    message_count = len(
        request.state.get(
            "messages",
            [],
        )
    )
    started_at = perf_counter()

    logger.info(
        "[model_call]调用开始 "
        "request_id=%s message_count=%d",
        request_id,
        message_count,
    )

    try:
        response = handler(request)
    except Exception as error:
        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000

        # 不记录消息正文，只保留问题编号、耗时和异常类型。
        logger.exception(
            "[model_call]调用失败 "
            "request_id=%s elapsed_ms=%.2f "
            "error_type=%s",
            request_id,
            elapsed_ms,
            type(error).__name__,
        )
        raise

    elapsed_ms = (
        perf_counter() - started_at
    ) * 1000

    logger.info(
        "[model_call]调用成功 "
        "request_id=%s elapsed_ms=%.2f",
        request_id,
        elapsed_ms,
    )

    return response


@dynamic_prompt
def report_prompt_switch(
    request: ModelRequest,
) -> str:
    """根据本次请求是否生成报告，选择对应系统Prompt。"""
    is_report = request.runtime.context.get(
        "report",
        False,
    )

    if is_report:
        return load_report_prompts()

    return load_system_prompts()
