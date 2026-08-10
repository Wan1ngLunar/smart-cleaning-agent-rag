import pytest
from langchain_core.documents import Document

from rag.hybrid_retriever import HybridRetriever


class StubVectorStoreService:
    """返回固定的向量检索结果并记录候选数量。"""

    def __init__(
        self,
        matches: list[tuple[Document, float]],
    ):
        self.matches = matches
        self.requested_k: int | None = None

    def search_with_relevance_scores(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        self.requested_k = k
        return self.matches[:k]


class StubBM25Retriever:
    """返回固定的BM25结果并记录候选数量。"""

    def __init__(
        self,
        matches: list[tuple[Document, float]],
    ):
        self.matches = matches
        self.requested_k: int | None = None

    def search(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        self.requested_k = k
        return self.matches[:k]


def build_document(
    document_id: str | None,
    content: str,
) -> Document:
    """创建带有指定ID的测试文档。"""
    return Document(
        id=document_id,
        page_content=content,
        metadata={"source": "测试资料.txt"},
    )


def test_search_fuses_and_deduplicates_two_rankings():
    """同时被两路召回的片段应去重并获得更高融合排名。"""
    shared_document = build_document(
        "shared",
        "低电量时自动回充。",
    )
    vector_only_document = build_document(
        "vector-only",
        "机器人可以自动导航。",
    )
    bm25_only_document = build_document(
        "bm25-only",
        "电量阈值为20%。",
    )

    vector_store = StubVectorStoreService(
        [
            (shared_document, 0.90),
            (vector_only_document, 0.80),
        ]
    )
    bm25_retriever = StubBM25Retriever(
        [
            (bm25_only_document, 4.20),
            (shared_document, 3.80),
        ]
    )

    retriever = HybridRetriever(
        vector_store_service=vector_store,
        bm25_retriever=bm25_retriever,
        vector_candidate_k=20,
        bm25_candidate_k=20,
        rrf_constant=60,
    )

    results = retriever.search(
        "低电量回充阈值",
        k=3,
    )

    assert [
        result.document.id
        for result in results
    ] == [
        "shared",
        "bm25-only",
        "vector-only",
    ]
    assert results[0].vector_rank == 1
    assert results[0].bm25_rank == 2
    assert results[0].vector_score == 0.90
    assert results[0].bm25_score == 3.80
    assert vector_store.requested_k == 20
    assert bm25_retriever.requested_k == 20


def test_search_respects_final_result_limit():
    """融合后的结果数量不应超过最终k。"""
    first_document = build_document(
        "first",
        "第一份资料",
    )
    second_document = build_document(
        "second",
        "第二份资料",
    )

    retriever = HybridRetriever(
        vector_store_service=StubVectorStoreService(
            [
                (first_document, 0.90),
                (second_document, 0.80),
            ]
        ),
        bm25_retriever=StubBM25Retriever([]),
    )

    results = retriever.search(
        "测试问题",
        k=1,
    )

    assert len(results) == 1


def test_search_requires_document_id():
    """缺少Chroma文档ID时无法安全合并两路结果。"""
    document_without_id = build_document(
        None,
        "没有ID的片段",
    )

    retriever = HybridRetriever(
        vector_store_service=StubVectorStoreService(
            [(document_without_id, 0.90)]
        ),
        bm25_retriever=StubBM25Retriever([]),
    )

    with pytest.raises(
        ValueError,
        match="每个文档都具有Chroma文档ID",
    ):
        retriever.search(
            "测试问题",
            k=3,
        )


@pytest.mark.parametrize(
    (
        "vector_candidate_k",
        "bm25_candidate_k",
        "rrf_constant",
    ),
    [
        (0, 20, 60),
        (20, 0, 60),
        (20, 20, 0),
    ],
)
def test_constructor_rejects_invalid_parameters(
    vector_candidate_k: int,
    bm25_candidate_k: int,
    rrf_constant: int,
):
    """候选数量和RRF常数都必须是正整数。"""
    with pytest.raises(ValueError):
        HybridRetriever(
            vector_store_service=(
                StubVectorStoreService([])
            ),
            bm25_retriever=StubBM25Retriever([]),
            vector_candidate_k=vector_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            rrf_constant=rrf_constant,
        )

def test_fuse_does_not_call_retrieval_services():
    """直接融合现有候选时不应再次执行向量或BM25检索。"""
    shared_document = build_document(
        "shared",
        "低电量时自动回充。",
    )

    vector_store = StubVectorStoreService([])
    bm25_retriever = StubBM25Retriever([])

    retriever = HybridRetriever(
        vector_store_service=vector_store,
        bm25_retriever=bm25_retriever,
    )

    results = retriever.fuse(
        vector_matches=[
            (shared_document, 0.90),
        ],
        bm25_matches=[
            (shared_document, 4.20),
        ],
        k=3,
    )

    assert len(results) == 1
    assert results[0].document.id == "shared"
    assert results[0].vector_rank == 1
    assert results[0].bm25_rank == 1

    # requested_k仍为None，证明两个Stub的search方法都没有被调用。
    assert vector_store.requested_k is None
    assert bm25_retriever.requested_k is None
