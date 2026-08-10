import pytest
from langchain_core.documents import Document

from rag.bm25_retriever import BM25Retriever


def build_documents() -> list[Document]:
    """创建不依赖真实Chroma数据库的测试文档。"""
    return [
        Document(
            id="battery",
            page_content="电量低于20%时，机器人会自动回充。",
            metadata={"source": "使用说明.txt"},
        ),
        Document(
            id="filter",
            page_content="滤网清洗后必须完全晾干再安装。",
            metadata={"source": "维护保养.txt"},
        ),
        Document(
            id="mop",
            page_content="拖布使用结束后需要清洗和晾晒。",
            metadata={"source": "维护保养.txt"},
        ),
    ]


def test_search_returns_most_relevant_document_first():
    """低电量回充问题应优先命中包含直接证据的文档。"""
    retriever = BM25Retriever(build_documents())

    results = retriever.search(
        "机器人低电量时为什么会自动回充？",
        k=3,
    )

    assert results
    assert results[0][0].id == "battery"


def test_search_respects_result_limit():
    """搜索结果数量不应超过调用方指定的k。"""
    retriever = BM25Retriever(build_documents())

    results = retriever.search(
        "清洗后如何晾干？",
        k=1,
    )

    assert len(results) == 1


def test_search_returns_empty_when_no_token_overlaps():
    """问题与所有文档都没有词元交集时应返回空列表。"""
    retriever = BM25Retriever(build_documents())

    results = retriever.search(
        "股票实时行情",
        k=3,
    )

    assert results == []


def test_search_handles_empty_documents_and_query():
    """空文档集合和空白问题都不应导致BM25异常。"""
    empty_retriever = BM25Retriever([])
    normal_retriever = BM25Retriever(
        build_documents()
    )

    assert empty_retriever.search("自动回充", k=3) == []
    assert normal_retriever.search("   ", k=3) == []


def test_search_rejects_invalid_result_limit():
    """k小于等于0属于调用错误，应提供明确异常。"""
    retriever = BM25Retriever(build_documents())

    with pytest.raises(
        ValueError,
        match="k必须是大于0的整数",
    ):
        retriever.search("自动回充", k=0)
