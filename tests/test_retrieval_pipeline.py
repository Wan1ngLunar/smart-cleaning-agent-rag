import logging

import pytest
from langchain_core.documents import Document

from rag.reranker import RerankerError, RerankResult
from rag.retrieval_pipeline import HybridRerankRetriever


class StubVectorStoreService:
    """使用内存数据替代真实Chroma，避免测试调用外部接口。"""

    def __init__(self, documents, vector_matches):
        self.documents = documents
        self.vector_matches = vector_matches
        self.requested_k = None

    def get_all_documents(self):
        """返回BM25建立索引所需的全部文档。"""
        return self.documents

    def search_with_relevance_scores(self, query, k):
        """模拟向量检索，并记录流水线传入的候选数量。"""
        self.requested_k = k
        return self.vector_matches[:k]


class StubReranker:
    """按照测试指定的文档ID模拟重排序结果。"""

    def __init__(self, selected_document_ids):
        self.selected_document_ids = selected_document_ids
        self.last_documents = []
        self.close_count = 0

    def rerank(self, query, documents, top_n):
        """返回指定顺序的重排序结果。"""
        self.last_documents = documents

        document_by_id = {
            document.id: document
            for document in documents
        }

        selected_documents = [
            document_by_id[document_id]
            for document_id in self.selected_document_ids
            if document_id in document_by_id
        ][:top_n]

        return [
            RerankResult(
                document=document,
                relevance_score=1.0 - rerank_rank * 0.01,
                original_rank=documents.index(document) + 1,
                rerank_rank=rerank_rank,
            )
            for rerank_rank, document in enumerate(
                selected_documents,
                start=1,
            )
        ]

    def close(self):
        """记录HTTP客户端被关闭的次数。"""
        self.close_count += 1


class FailingReranker(StubReranker):
    """模拟重排序接口暂时不可用。"""

    def rerank(self, query, documents, top_n):
        self.last_documents = documents
        raise RerankerError("模拟重排序服务不可用")


def build_document(document_id, content):
    """创建带有稳定ID的测试文档。"""
    return Document(
        id=document_id,
        page_content=content,
        metadata={"source": f"{document_id}.txt"},
    )


def test_bm25_can_recover_evidence_filtered_by_vector_score():
    """验证向量分数较低时，BM25仍能召回答案证据。"""
    direct_evidence = build_document(
        "direct",
        "机器人剩余电量低于20%时会优先返回充电座。",
    )
    unrelated = build_document(
        "unrelated",
        "量子计算机使用量子比特执行运算。",
    )

    vector_store = StubVectorStoreService(
        documents=[direct_evidence, unrelated],
        # 该分数低于阈值，因此不会进入向量候选。
        vector_matches=[(unrelated, 0.10)],
    )
    reranker = StubReranker(["direct"])

    pipeline = HybridRerankRetriever(
        vector_store_service=vector_store,
        reranker=reranker,
    )

    results = pipeline.retrieve(
        "机器人电量低于多少时会优先回充？"
    )

    assert [document.id for document in results] == ["direct"]
    assert vector_store.requested_k == 10


def test_pipeline_returns_documents_in_reranker_order():
    """验证最终结果严格使用重排序器给出的顺序。"""
    first = build_document(
        "first",
        "HEPA滤网需要定期维护。",
    )
    complete = build_document(
        "complete",
        "HEPA滤网每周清理，每1至2个月水洗，3至6个月更换。",
    )

    vector_store = StubVectorStoreService(
        documents=[first, complete],
        vector_matches=[
            (first, 0.80),
            (complete, 0.75),
        ],
    )
    reranker = StubReranker(["complete", "first"])

    pipeline = HybridRerankRetriever(
        vector_store_service=vector_store,
        reranker=reranker,
    )

    results = pipeline.retrieve("HEPA滤网应该怎样维护？")

    assert [document.id for document in results] == [
        "complete",
        "first",
    ]


def test_pipeline_falls_back_to_hybrid_results(
    caplog,
):
    """验证重排序接口失败时会安全降级，不会中断RAG。"""
    documents = [
        build_document(
            f"document-{index}",
            f"扫地机器人滤网维护说明{index}",
        )
        for index in range(1, 5)
    ]

    vector_store = StubVectorStoreService(
        documents=documents,
        vector_matches=[
            (document, 0.90 - index * 0.01)
            for index, document in enumerate(documents)
        ],
    )
    reranker = FailingReranker([])

    pipeline = HybridRerankRetriever(
        vector_store_service=vector_store,
        reranker=reranker,
    )

    with caplog.at_level(logging.WARNING):
        results = pipeline.retrieve("滤网维护")

    expected_documents = reranker.last_documents[:3]

    assert results == expected_documents
    assert "已降级为RRF混合检索结果" in caplog.text


def test_pipeline_closes_reranker_only_once():
    """验证重复关闭流水线不会重复释放同一资源。"""
    document = build_document(
        "document",
        "扫地机器人维护说明。",
    )
    vector_store = StubVectorStoreService(
        documents=[document],
        vector_matches=[(document, 0.90)],
    )
    reranker = StubReranker(["document"])

    pipeline = HybridRerankRetriever(
        vector_store_service=vector_store,
        reranker=reranker,
    )

    pipeline.close()
    pipeline.close()

    assert reranker.close_count == 1

    with pytest.raises(
        RuntimeError,
        match="检索流水线已经关闭",
    ):
        pipeline.retrieve("维护方法")
