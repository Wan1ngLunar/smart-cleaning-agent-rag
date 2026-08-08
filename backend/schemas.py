from typing import Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
)

"""
定义前后端 HTTP 接口的数据格式，自动做参数校验。
前端发 JSON 给后端、后端返回 JSON 给前端，全部由这些类约束。
父类 ApiModel 是所有模型的公共基础，统一全局校验规则。
"""

class ApiModel(BaseModel):
    """为所有API数据模型提供统一的严格校验规则。"""

    model_config = ConfigDict(
        # 自动去除字符串首尾空白，避免只输入空格绕过校验。
        str_strip_whitespace=True,
        # 拒绝接口契约之外的字段，及时发现前后端字段拼写错误。
        extra="forbid",
    )


class HealthResponse(ApiModel):
    """描述后端健康检查的响应。"""

    status: Literal["ok"] = "ok"


class ChatRequest(ApiModel):
    """描述前端发送给聊天接口的请求。"""

    # 只接受UUID第4版，与项目当前生成的会话ID保持一致。
    thread_id: UUID4

    # 限制问题长度，避免空问题和异常大请求占用模型资源。
    message: str = Field(
        min_length=1,
        max_length=4000,
    )


class ChatResponse(ApiModel):
    """描述聊天接口成功完成后的响应。"""

    thread_id: UUID4
    answer: str = Field(
        min_length=1,
    )


class HistoryMessage(ApiModel):
    """描述一条允许展示给前端的历史消息。"""

    # API只暴露用户消息和最终助手消息，不暴露工具内部状态。
    role: Literal[
        "user",
        "assistant",
    ]
    content: str = Field(
        min_length=1,
    )


class HistoryResponse(ApiModel):
    """描述指定会话的可展示历史记录。"""

    thread_id: UUID4

    # 每个响应使用独立列表，避免不同请求意外共享可变默认值。
    messages: list[HistoryMessage] = Field(
        default_factory=list,
    )


class ErrorResponse(ApiModel):
    """描述后端返回给前端的安全错误。"""

    detail: str = Field(
        min_length=1,
    )

    # Agent执行失败时返回12位问题编号，参数校验错误时可以为空。
    request_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{12}$",
    )
