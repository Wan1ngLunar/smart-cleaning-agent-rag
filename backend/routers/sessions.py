from fastapi import APIRouter
from pydantic import UUID4

from backend.dependencies import AgentDependency
from backend.schemas import HistoryResponse

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["会话"],
)


@router.get(
    "/{thread_id}/messages",
    response_model=HistoryResponse,
    summary="读取指定会话的公开历史消息",
)
def get_session_history(
    thread_id: UUID4,
    agent: AgentDependency,
) -> HistoryResponse:
    """读取用户和最终助手消息，不暴露工具内部状态。"""
    # ReactAgent使用字符串形式的thread_id访问LangGraph状态。
    messages = agent.get_history(
        str(thread_id)
    )

    # HistoryResponse会再次校验角色、正文和UUID格式。
    return HistoryResponse(
        thread_id=thread_id,
        messages=messages,
    )
