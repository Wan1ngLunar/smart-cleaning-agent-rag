from typing import Any

import httpx

"""这是Streamlit 前端侧的工具类，专门统一调用后端 /api/v1/chat、会话历史接口，做了：
统一 HTTP 超时、请求头、地址处理
统一捕获网络 / 服务报错，包装成安全友好提示，不把底层堆栈抛给页面
严格校验后端返回 JSON 格式，格式不对直接抛业务异常
支持测试注入 MockTransport，不用真实网络请求做单元测试"""
# 聊天模型可能需要较长时间，因此总超时设置为120秒。
DEFAULT_TIMEOUT_SECONDS = 120.0

# 建立TCP连接不应等待太久，单独设置10秒连接超时。
CONNECT_TIMEOUT_SECONDS = 10.0


class ApiClientError(RuntimeError):
    """表示前端无法安全完成后端API请求。"""

    def __init__(self, public_message: str):
        # 页面只能展示安全说明，不能显示HTTPX内部连接栈。
        self.public_message = public_message
        super().__init__(public_message)


class BackendApiClient:
    """封装Streamlit对FastAPI后端的HTTP调用。"""

    def __init__(
        self,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
    ):
        # 统一去除空格和末尾斜杠，避免路径拼接出现双斜杠。
        normalized_base_url = (
            base_url.strip().rstrip("/")
        )

        if not normalized_base_url:
            raise ValueError(
                "FastAPI后端地址不能为空"
            )

        self._client = httpx.Client(
            base_url=normalized_base_url,
            timeout=httpx.Timeout(
                DEFAULT_TIMEOUT_SECONDS,
                connect=CONNECT_TIMEOUT_SECONDS,
            ),
            headers={
                "Accept": "application/json",
            },
            # 生产环境不传transport，测试使用MockTransport避免真实网络。
            transport=transport,
        )

    def close(self) -> None:
        """关闭HTTP连接池；重复关闭不会报错。"""
        self._client.close()

    @staticmethod
    def _get_error_detail(
        response: httpx.Response,
    ) -> str:
        """从错误响应中提取后端允许公开的安全说明。"""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")

            if (
                isinstance(detail, str)
                and detail.strip()
            ):
                return detail.strip()

        # 不返回原始响应正文，避免HTML错误页或内部信息进入页面。
        return (
            "后端请求失败，"
            f"HTTP状态码：{response.status_code}"
        )

    def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送请求，并把成功响应校验为JSON对象。"""
        try:
            response = self._client.request(
                method,
                path,
                **kwargs,
            )
        except httpx.RequestError as error:
            # 保留异常因果链供开发排查，页面只读取安全说明。
            raise ApiClientError(
                "无法连接后端服务，请确认FastAPI已经启动。"
            ) from error

        if response.is_error:
            raise ApiClientError(
                self._get_error_detail(
                    response
                )
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise ApiClientError(
                "后端返回了无法解析的JSON数据。"
            ) from error

        if not isinstance(payload, dict):
            raise ApiClientError(
                "后端响应格式不符合接口约定。"
            )

        return payload

    def get_history(
        self,
        thread_id: str,
    ) -> list[dict[str, str]]:
        """读取指定会话允许展示的历史消息。"""
        payload = self._request_json(
            "GET",
            (
                f"/api/v1/sessions/"
                f"{thread_id}/messages"
            ),
        )
        raw_messages = payload.get(
            "messages"
        )

        if not isinstance(raw_messages, list):
            raise ApiClientError(
                "后端历史响应缺少消息列表。"
            )

        messages: list[dict[str, str]] = []

        for raw_message in raw_messages:
            if not isinstance(
                raw_message,
                dict,
            ):
                raise ApiClientError(
                    "后端历史消息格式不正确。"
                )

            role = raw_message.get("role")
            content = raw_message.get(
                "content"
            )

            if (
                role not in {
                    "user",
                    "assistant",
                }
                or not isinstance(
                    content,
                    str,
                )
                or not content.strip()
            ):
                raise ApiClientError(
                    "后端历史消息内容不符合约定。"
                )

            messages.append(
                {
                    "role": role,
                    "content": content.strip(),
                }
            )

        return messages

    def chat(
        self,
        thread_id: str,
        message: str,
    ) -> str:
        """向指定会话发送问题并返回完整回答。"""
        payload = self._request_json(
            "POST",
            "/api/v1/chat",
            json={
                "thread_id": thread_id,
                "message": message,
            },
        )
        answer = payload.get("answer")

        if (
            not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ApiClientError(
                "后端聊天响应缺少有效回答。"
            )

        return answer.strip()
