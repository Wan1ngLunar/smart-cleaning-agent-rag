import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document

from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path

# 使用问答检索任务，让模型优先判断片段能否直接回答问题。
DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant "
    "passages that answer the query."
)


class RerankerError(RuntimeError):
    """表示重排序服务连接失败或返回了无效数据。"""


@dataclass(frozen=True)
class RerankResult:
    """保存重排序后的文档、分数和排名诊断信息。"""

    document: Document
    relevance_score: float # 重排返回 0‑1 相关性分数，越高越相关
    original_rank: int
    rerank_rank: int


class QwenReranker:
    """通过HTTPX调用百炼qwen3-rerank接口。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "qwen3-rerank",
        instruct: str = DEFAULT_RERANK_INSTRUCTION, # 传给 rerank 模型的任务提示词
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None, # httpx 底层传输层，单元测试的时候传入 mock transport，不用真实发网络请求。
    ):
        normalized_api_key = api_key.strip()
        normalized_base_url = (
            base_url.strip().rstrip("/")
            + "/"
        )
        normalized_model_name = model_name.strip()

        if not normalized_api_key:
            raise ValueError("重排序API Key不能为空")

        if not normalized_model_name:
            raise ValueError("重排序模型名称不能为空")

        if timeout_seconds <= 0:
            raise ValueError(
                "重排序超时时间必须大于0"
            )

        parsed_base_url = httpx.URL(
            normalized_base_url
        ) # httpx.URL()解析 base_url，强制校验必须是 http/https，防止传错地址。

        if (
            parsed_base_url.scheme
            not in {"http", "https"}
            or not parsed_base_url.host
        ):
            raise ValueError(
                "重排序接口地址必须是有效的HTTP或HTTPS地址"
            )

        self.model_name = normalized_model_name
        self.instruct = instruct.strip()

        # base_url保留结尾斜杠，后续使用相对路径reranks时不会丢失v1。
        self._client = httpx.Client(
            base_url=normalized_base_url,
            headers={
                "Authorization": (
                    f"Bearer {normalized_api_key}"
                ),
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
            transport=transport,
        )

    # __enter__ 和 __exit__ 是 Python 上下文管理器协议：
    # 进入with调用__enter__拿到对象；
    # 退出with（正常结束 / 抛异常）一定调用__exit__；
    # 这里__exit__调用close()，把 httpx 连接池关掉，避免资源泄漏。
    def __enter__(self) -> Self:
        """支持使用with语句自动释放HTTP连接池。"""
        return self

    # 这三个参数用来接收with 块内部抛出的异常信息。本代码不处理异常，只是单纯执行关闭：无论有没有报错，都执行 self.close() 释放 httpx 连接池。
    def __exit__(
        self,
        exception_type: type[BaseException] | None, # 异常的类型，例如 RerankerError；没有异常就是None
        exception: BaseException | None, # 异常对象本身；没有异常就是None
        traceback: TracebackType | None,# 堆栈追踪对象，打印报错栈用；没有异常就是None
    ) -> None:
        self.close()

    def close(self) -> None:
        """关闭HTTPX连接池。"""
        self._client.close()

    @staticmethod
    def _get_request_id(
        response: httpx.Response,
    ) -> str | None:
        """从错误响应中提取可用于排查的问题编号。"""
        try:
            response_data = response.json()
        except ValueError:
            response_data = {}

        if isinstance(response_data, dict):
            request_id = response_data.get(
                "request_id"
            )

            if request_id:
                return str(request_id)

        header_request_id = response.headers.get(
            "x-request-id"
        )

        return header_request_id or None

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int,
    ) -> list[RerankResult]:
        """按照候选片段回答问题的能力重新排序。"""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("重排序问题不能为空")

        if top_n <= 0:
            raise ValueError(
                "top_n必须是大于0的整数"
            )

        # 没有候选时直接返回，避免产生无意义的远程请求。
        if not documents:
            return []

        requested_top_n = min(
            top_n,
            len(documents),
        ) # 如果只传入 2 篇文档，top_n 传 5，就实际请求 2 条，避免接口报错。

        request_body = {
            "model": self.model_name,
            "query": normalized_query,
            "documents": [
                document.page_content
                for document in documents
            ],
            "top_n": requested_top_n,
        }

        # instruct为空时使用模型默认的问答检索任务。
        if self.instruct:
            request_body[
                "instruct"
            ] = self.instruct

        try:
            # 这里必须使用相对路径，保留base_url中的compatible-api/v1。
            response = self._client.post(
                "reranks",
                json=request_body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            request_id = self._get_request_id(
                error.response
            )
            request_id_text = (
                f"，问题编号：{request_id}"
                if request_id
                else ""
            )

            # 不记录响应正文，避免未来接口错误意外包含敏感内容。
            raise RerankerError(
                "重排序服务返回"
                f"HTTP {error.response.status_code}"
                f"{request_id_text}"
            ) from error
        except httpx.RequestError as error:
            raise RerankerError(
                "无法连接重排序服务"
            ) from error

        try:
            response_data = response.json()
        except ValueError as error:
            raise RerankerError(
                "重排序服务返回了非JSON数据"
            ) from error

        raw_results = (
            response_data.get("results")
            if isinstance(response_data, dict)
            else None
        )

        if (
            not isinstance(raw_results, list)
            or not raw_results
            or len(raw_results) > requested_top_n
        ):
            raise RerankerError(
                "重排序服务返回的results无效"
            )

        seen_indices: set[int] = set()
        parsed_results: list[RerankResult] = []

        for rerank_rank, raw_result in enumerate(
            raw_results,
            start=1,
        ):
            if not isinstance(raw_result, dict):
                raise RerankerError(
                    "重排序结果格式无效"
                )

            document_index = raw_result.get(
                "index"
            ) # 输入 documents 数组的下标，从 0 开始。
            relevance_score = raw_result.get(
                "relevance_score"
            )

            if (
                not isinstance(document_index, int)
                or isinstance(document_index, bool)
                or document_index < 0
                or document_index >= len(documents)
                or document_index in seen_indices
            ):
                raise RerankerError(
                    "重排序结果中的文档下标无效"
                )

            if (
                not isinstance(
                    relevance_score,
                    (int, float),
                )
                or isinstance(
                    relevance_score,
                    bool,
                )
                or not 0
                <= float(relevance_score)
                <= 1
            ):
                raise RerankerError(
                    "重排序结果中的相关性分数无效"
                )

            seen_indices.add(document_index)
            parsed_results.append(
                RerankResult(
                    document=documents[
                        document_index
                    ],
                    relevance_score=float(
                        relevance_score
                    ),
                    # 输入候选下标从0开始，人类可读排名从1开始。
                    original_rank=(
                        document_index + 1
                    ),
                    rerank_rank=rerank_rank,
                )
            )

        return parsed_results

def build_reranker(
    environment: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> QwenReranker:
    """根据安全环境变量和公开YAML配置创建重排序客户端。"""
    if environment is None:
        # 显式指定.env路径，避免标准输入和不同启动目录下自动查找失败。
        # override=False保证系统环境变量优先于本地.env。
        load_dotenv(
            dotenv_path=get_abs_path(".env"),
            override=False,
        )
        selected_environment: Mapping[
            str,
            str,
        ] = os.environ
    else:
        # 单元测试传入内存字典时，不读取开发者真实.env。
        selected_environment = environment

    api_key = str(
        selected_environment.get(
            "DASHSCOPE_API_KEY",
            "",
        )
    ).strip()
    base_url = str(
        selected_environment.get(
            "DASHSCOPE_RERANK_BASE_URL",
            "",
        )
    ).strip()

    if not api_key:
        raise RuntimeError(
            "缺少DASHSCOPE_API_KEY，"
            "无法创建重排序客户端"
        )

    if not base_url:
        raise RuntimeError(
            "缺少DASHSCOPE_RERANK_BASE_URL，"
            "无法创建重排序客户端"
        )

    rerank_config = rag_conf.get("rerank")

    if not isinstance(rerank_config, dict):
        raise RuntimeError(
            "config/rag.yml缺少有效rerank配置"
        )

    try:
        model_name = str(
            rerank_config["model_name"]
        ).strip()
        instruct = str(
            rerank_config["instruct"]
        ).strip()
        timeout_seconds = float(
            rerank_config["timeout_seconds"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "config/rag.yml中的rerank配置无效"
        ) from error

    return QwenReranker(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        instruct=instruct,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
