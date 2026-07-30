from langchain_core.documents import Document

from rag.rag_service import (
    EMPTY_ANSWER_RESPONSE,
    NO_CONTEXT_RESPONSE,
    RagSummarizeService,
)


class StubRetriever:
    """返回预设文档，避免测试依赖真实Chroma数据库。"""

    def __init__(self, documents: list[Document]):
        self.documents = documents

    def invoke(self, query: str) -> list[Document]:
        # 保留query参数以模拟LangChain Retriever接口。
        return self.documents


class StubChain:
    """记录模型输入并返回预设文本，避免测试调用DashScope API。"""

    def __init__(self, answer: str):
        self.answer = answer
        self.last_payload: dict[str, str] | None = None

    def invoke(self, payload: dict[str, str]) -> str:
        self.last_payload = payload
        return self.answer


def build_service(
    documents: list[Document],
    answer: str = "建议定期清理滚刷。[1]",
) -> tuple[RagSummarizeService, StubChain]:
    """绕过生产初始化，组装只包含假Retriever和假Chain的测试服务。"""
    service = RagSummarizeService.__new__(RagSummarizeService)
    chain = StubChain(answer)
    service.retriever = StubRetriever(documents)
    service.chain = chain
    return service, chain


def test_rag_answer_appends_deduplicated_sources_with_pdf_page():
    """回答应附加去重来源，并正确展示PDF的人类可读页码。"""
    documents = [
        Document(
            page_content="滚刷缠绕毛发后会影响清洁效率。",
            metadata={
                "source": r"data\扫地机器人100问.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content="建议定期清理滚刷上的毛发。",
            metadata={
                "source": r"data\扫地机器人100问.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content="滤网需要保持干燥。",
            metadata={"source": "data/维护保养.txt"},
        ),
    ]
    service, chain = build_service(documents)

    result = service.rag_summarize("如何维护滚刷？")

    assert "建议定期清理滚刷。[1]" in result
    assert result.count("[1] 扫地机器人100问.pdf（第1页）") == 1
    assert result.count("[2] 维护保养.txt") == 1
    assert chain.last_payload is not None
    assert "来源[1]" in chain.last_payload["context"]
    assert "来源[2]" in chain.last_payload["context"]


def test_rag_does_not_call_model_without_valid_documents():
    """检索为空或只有空白内容时，应直接返回空结果而不是让模型编造。"""
    documents = [
        Document(
            page_content="   ",
            metadata={"source": "data/空文档.txt"},
        )
    ]
    service, chain = build_service(documents)

    result = service.rag_summarize("知识库中不存在的问题")

    assert result == NO_CONTEXT_RESPONSE
    assert chain.last_payload is None


def test_rag_keeps_sources_when_model_returns_empty_answer():
    """模型返回空文本时仍应说明异常，并保留已检索到的来源。"""
    documents = [
        Document(
            page_content="边刷需要定期检查。",
            metadata={"source": "data/维护保养.txt"},
        )
    ]
    service, _ = build_service(documents, answer="   ")

    result = service.rag_summarize("如何检查边刷？")

    assert EMPTY_ANSWER_RESPONSE in result
    assert "[1] 维护保养.txt" in result
