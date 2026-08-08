from fastapi import (
    APIRouter,
    status,
)
from fastapi.responses import JSONResponse

from agent.react_agent import AgentExecutionError
from backend.dependencies import AgentDependency
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["聊天"],
)

# Agent正常执行但没有产生文本时，向前端返回固定安全说明。
EMPTY_AGENT_RESPONSE_DETAIL = (
    "Agent没有返回可展示的文本，请稍后重试。"
)


def create_error_response(
    detail: str,
    request_id: str | None = None,
) -> JSONResponse:
    """创建符合统一错误契约的502响应。"""
    payload = ErrorResponse(
        detail=detail,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=payload.model_dump(
            mode="json",
        ),
    )


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "Agent或上游模型执行失败",
        },
    },
    summary="向指定会话发送问题",
)
def chat(
    request: ChatRequest,
    agent: AgentDependency,
) -> ChatResponse | JSONResponse:
    """执行Agent问答，并返回最后一条有效回答。"""
    try:
        # execute_stream可能产生多个LangGraph阶段结果。
        response_chunks = list(
            agent.execute_stream(
                request.message,
                thread_id=str(
                    request.thread_id
                ),
            )
        )
    except AgentExecutionError as error:
        # Agent已经记录真实异常堆栈，API只返回安全说明和问题编号。
        return create_error_response(
            detail=error.public_message,
            request_id=error.request_id,
        )

    # 从后向前寻找最后一条非空文本，避免空白片段成为最终回答。
    answer = next(
        (
            chunk.strip()
            for chunk in reversed(
                response_chunks
            )
            if chunk.strip()
        ),
        "",
    ) # Agent 运行会输出多段中间文本，只把最后一段真正有内容的回答返回给前端。

    if not answer:
        return create_error_response(
            detail=EMPTY_AGENT_RESPONSE_DETAIL,
        )

    return ChatResponse(
        thread_id=request.thread_id,
        answer=answer,
    )
