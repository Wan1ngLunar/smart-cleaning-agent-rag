from langchain_core.documents import Document

from rag.rag_service import (
    EMPTY_ANSWER_RESPONSE,
    INSUFFICIENT_CONTEXT_MARKER,
    NO_CONTEXT_RESPONSE,
    RagSummarizeService,
)
from utils.prompt_loader import load_rag_prompts


class StubRetriever:
    """返回预设文档，避免测试调用真实检索和重排序服务。"""

    def __init__(
        self,
        documents: list[Document],
    ):
        self.documents = documents
        self.last_query: str | None = None
        self.close_count = 0

    def retrieve(
        self,
        query: str,
    ) -> list[Document]:
        """记录问题并返回预设的最终检索结果。"""
        self.last_query = query
        return self.documents

    def close(self) -> None:
        """记录资源关闭次数。"""
        self.close_count += 1


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
    """绕过生产初始化，组装使用预设文档和回答的测试服务。"""
    service = RagSummarizeService.__new__(
        RagSummarizeService
    )
    chain = StubChain(answer)

    # 测试使用假检索器，不访问Chroma、BM25或外部重排序接口。
    service.retriever = StubRetriever(documents)
    service._closed = False
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

def test_rag_does_not_call_model_when_retriever_returns_empty():
    """检索流水线没有返回文档时，服务应直接拒答。"""
    service, chain = build_service([])

    result = service.rag_summarize(
        "知识库无法回答的问题"
    )

    assert result == NO_CONTEXT_RESPONSE
    assert chain.last_payload is None

def test_rag_rejects_when_model_reports_insufficient_context():
    """高分片段不能直接回答问题时，不应展示答案或参考来源。"""
    documents = [
        Document(
            page_content="扫地机器人滤网堵塞时需要及时清理。",
            metadata={"source": "data/故障排除.txt"},
        )
    ]
    service, chain = build_service(
        documents,
        answer=INSUFFICIENT_CONTEXT_MARKER,
    )

    result = service.rag_summarize(
        "戴森手持吸尘器应该拆洗哪个部件？"
    )

    assert result == NO_CONTEXT_RESPONSE

    # last_payload不为None，证明模型执行了资料充分性判断。
    assert chain.last_payload is not None

    # 范围不匹配时不能附加看似可信但实际无关的来源。
    assert "参考来源" not in result
    assert "故障排除.txt" not in result

def test_rag_prompt_contains_insufficient_context_marker():
    """Prompt必须包含服务层识别的资料不足标记。"""
    prompt_text = load_rag_prompts()

    assert INSUFFICIENT_CONTEXT_MARKER in prompt_text

def test_rag_service_closes_retriever_only_once():
    """重复关闭RAG服务时不应重复关闭检索流水线。"""
    service, _ = build_service([])

    service.close()
    service.close()

    assert service.retriever.close_count == 1

def test_rag_scope_rules_are_sent_as_system_message():
    """知识库边界必须使用System消息，不能与用户输入混为普通文本。"""
    service = RagSummarizeService.__new__(
        RagSummarizeService
    )
    service.prompt_text = load_rag_prompts()

    prompt_template = service._build_prompt_template()
    prompt_value = prompt_template.invoke(
        {
            "input": "戴森手持吸尘器应该清洗什么部件？",
            "context": "扫地机器人需要清理滤网和滚刷。",
        }
    )
    messages = prompt_value.to_messages()

    # 第一条必须是System消息，并包含明确的知识库设备边界。
    assert messages[0].type == "system"
    assert "只覆盖扫地机器人和扫拖一体机器人" in messages[0].content
    assert "禁止把扫地机器人的处理建议迁移到其他设备" in messages[0].content

    # 用户问题和检索资料应放在后面的Human消息中。
    assert messages[1].type == "human"
    assert "戴森手持吸尘器" in messages[1].content
    assert "扫地机器人需要清理滤网和滚刷" in messages[1].content
