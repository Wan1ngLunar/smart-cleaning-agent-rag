import logging

import pytest
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import (
    ToolCallRequest,
)
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime

from agent.tools.middleware import (
    monitor_model,
    monitor_tool,
)


def create_tool_request(
    tool_name: str,
    private_argument: str,
    request_id: str,
) -> tuple[
    ToolCallRequest,
    dict[str, object],
]:
    """创建不执行真实工具的测试请求。"""
    runtime_context: dict[str, object] = {
        "request_id": request_id,
        "report": False,
    }
    runtime = Runtime(
        context=runtime_context
    )

    request = ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {
                "query": private_argument,
            },
            "id": "test-tool-call",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=runtime,
    )

    return request, runtime_context


def create_model_request(
    private_message: str,
    request_id: str,
) -> ModelRequest:
    """创建使用LangChain官方假模型的测试请求。"""
    message = HumanMessage(
        content=private_message
    )
    runtime = Runtime(
        context={
            "request_id": request_id,
            "report": False,
        }
    )

    return ModelRequest(
        # 假模型只满足类型要求，测试处理函数不会真正调用它。
        model=FakeListChatModel(
            responses=["未使用的响应"]
        ),
        messages=[message],
        state={
            "messages": [
                message,
            ]
        },
        runtime=runtime,
    )


def test_monitor_tool_logs_success_without_arguments(
    caplog: pytest.LogCaptureFixture,
):
    """工具成功日志应有编号和耗时，但不能包含工具参数。"""
    private_argument = "不应写入日志的工具参数"
    request, runtime_context = (
        create_tool_request(
            "fill_context_for_report",
            private_argument,
            "tool-success-id",
        )
    )

    def successful_handler(
        current_request: ToolCallRequest,
    ) -> ToolMessage:
        """返回固定工具结果，不执行真实业务函数。"""
        return ToolMessage(
            content="工具执行成功",
            tool_call_id=current_request.tool_call[
                "id"
            ],
        )

    with caplog.at_level(
        logging.INFO,
        logger="agent",
    ):
        result = monitor_tool.wrap_tool_call(
            request,
            successful_handler,
        )

    assert result.content == "工具执行成功"
    assert runtime_context["report"] is True
    assert "request_id=tool-success-id" in (
        caplog.text
    )
    assert "tool=fill_context_for_report" in (
        caplog.text
    )
    assert "调用成功" in caplog.text
    assert "elapsed_ms=" in caplog.text

    # 工具参数可能包含用户问题或身份信息，不能写入日志。
    assert private_argument not in caplog.text


def test_monitor_tool_logs_failure_and_reraises(
    caplog: pytest.LogCaptureFixture,
):
    """工具失败应记录异常类型，并重新抛出原始异常。"""
    private_argument = "失败时也不能记录的参数"
    request, _ = create_tool_request(
        "rag_summarize",
        private_argument,
        "tool-failure-id",
    )

    def failing_handler(
        current_request: ToolCallRequest,
    ) -> ToolMessage:
        """模拟工具内部发生异常。"""
        raise RuntimeError(
            "模拟工具异常"
        )

    with caplog.at_level(
        logging.ERROR,
        logger="agent",
    ):
        with pytest.raises(
            RuntimeError,
            match="模拟工具异常",
        ):
            monitor_tool.wrap_tool_call(
                request,
                failing_handler,
            )

    assert "request_id=tool-failure-id" in (
        caplog.text
    )
    assert "error_type=RuntimeError" in (
        caplog.text
    )
    assert "elapsed_ms=" in caplog.text
    assert private_argument not in caplog.text


def test_monitor_model_logs_success_without_message(
    caplog: pytest.LogCaptureFixture,
):
    """模型成功日志应记录消息数，但不能记录消息正文。"""
    private_message = "不应写入日志的用户消息"
    request = create_model_request(
        private_message,
        "model-success-id",
    )

    def successful_handler(
        current_request: ModelRequest,
    ) -> ModelResponse:
        """返回固定模型响应，不调用真实模型。"""
        return ModelResponse(
            result=[
                AIMessage(
                    content="固定测试回答"
                )
            ]
        )

    with caplog.at_level(
        logging.INFO,
        logger="agent",
    ):
        response = (
            monitor_model.wrap_model_call(
                request,
                successful_handler,
            )
        )

    assert response.result[0].content == (
        "固定测试回答"
    )
    assert "request_id=model-success-id" in (
        caplog.text
    )
    assert "message_count=1" in caplog.text
    assert "调用成功" in caplog.text
    assert "elapsed_ms=" in caplog.text

    # 日志只记录消息数量，不保存Prompt或用户正文。
    assert private_message not in caplog.text


def test_monitor_model_logs_failure_and_reraises(
    caplog: pytest.LogCaptureFixture,
):
    """模型失败应记录异常类型，并交给Agent请求边界处理。"""
    request = create_model_request(
        "模型失败测试消息",
        "model-failure-id",
    )

    def failing_handler(
        current_request: ModelRequest,
    ) -> ModelResponse:
        """模拟模型网络连接失败。"""
        raise ConnectionError(
            "模拟模型连接失败"
        )

    with caplog.at_level(
        logging.ERROR,
        logger="agent",
    ):
        with pytest.raises(
            ConnectionError,
            match="模拟模型连接失败",
        ):
            monitor_model.wrap_model_call(
                request,
                failing_handler,
            )

    assert "request_id=model-failure-id" in (
        caplog.text
    )
    assert "error_type=ConnectionError" in (
        caplog.text
    )
    assert "elapsed_ms=" in caplog.text
