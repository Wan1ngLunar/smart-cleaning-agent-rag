from langchain_core.documents import Document

from rag.rag_service import (
    EMPTY_ANSWER_RESPONSE,
    INSUFFICIENT_CONTEXT_MARKER,
    NO_CONTEXT_RESPONSE,
    RagSummarizeService,
)
from utils.prompt_loader import load_rag_prompts


class StubVectorStore:
    """返回预设的文档和分数，避免测试依赖真实Chroma数据库。"""

    def __init__(
        self,
        scored_documents: list[tuple[Document, float]],
    ):
        self.scored_documents = scored_documents

    def search_with_relevance_scores(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        # 保留query参数，用于模拟真实向量存储接口。
        return self.scored_documents


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
    relevance_scores: list[float] | None = None,
) -> tuple[RagSummarizeService, StubChain]:
    """绕过生产初始化，组装使用预设文档、分数和回答的测试服务。"""
    service = RagSummarizeService.__new__(
        RagSummarizeService
    )
    chain = StubChain(answer)

    # 没有指定分数时使用0.9，代表测试文档具有较高相关性。
    if relevance_scores is None:
        relevance_scores = [0.9] * len(documents)

    # 文档和分数必须一一对应，数量不一致说明测试数据写错。
    if len(documents) != len(relevance_scores):
        raise ValueError("测试文档数量必须与分数数量一致")

    scored_documents = list(
        zip(
            documents,
            relevance_scores,
            strict=True,
        )
    )

    service.vector_store = StubVectorStore(
        scored_documents
    )
    service.min_relevance_score = 0.20
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

def test_rag_does_not_call_model_when_all_documents_score_too_low():
    """所有片段低于最低分时，应直接拒答且不调用模型。"""
    documents = [
        Document(
            page_content="这是一段与用户问题无关的知识内容。",
            metadata={"source": "data/无关资料.txt"},
        )
    ]
    service, chain = build_service(
        documents,
        relevance_scores=[0.10],
    )

    result = service.rag_summarize(
        "番茄炒蛋应该放多少糖？"
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
        relevance_scores=[0.80],
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
