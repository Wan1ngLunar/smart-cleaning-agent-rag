from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from rag.bm25_retriever import BM25Retriever
from rag.vector_store import VectorStoreService


@dataclass(frozen=True)
class HybridSearchResult:
    """保存混合检索结果及两路检索的诊断信息。"""

    document: Document
    fusion_score: float
    vector_rank: int | None
    bm25_rank: int | None
    vector_score: float | None
    bm25_score: float | None


@dataclass
class _FusionCandidate:
    """在RRF计算过程中保存可更新的候选状态。"""

    document: Document
    fusion_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None


class HybridRetriever:
    """使用RRF融合向量检索与BM25关键词检索排名。"""

    def __init__(
        self,
        vector_store_service: VectorStoreService,
        bm25_retriever: BM25Retriever,
        vector_candidate_k: int = 20, # 向量先捞20条候选
        bm25_candidate_k: int = 20, # BM25先捞20条候选
        rrf_constant: int = 60, # RRF公式常数，一般取60
    ):
        if vector_candidate_k <= 0:
            raise ValueError(
                "vector_candidate_k必须是大于0的整数"
            )

        if bm25_candidate_k <= 0:
            raise ValueError(
                "bm25_candidate_k必须是大于0的整数"
            )

        if rrf_constant <= 0:
            raise ValueError(
                "rrf_constant必须是大于0的整数"
            )

        self.vector_store_service = (
            vector_store_service
        )
        self.bm25_retriever = bm25_retriever
        self.vector_candidate_k = vector_candidate_k
        self.bm25_candidate_k = bm25_candidate_k
        self.rrf_constant = rrf_constant

    # 向量检索、BM25 检索返回的 Document，靠 document.id 判断是不是同一条知识库片段。
    # 如果没有 id，就分不清向量返回的文档和 BM25 返回的文档是不是同一个，RRF 融合就会重复计算。
    @staticmethod
    def _get_document_id(
        document: Document,
    ) -> str:
        """读取用于跨检索渠道去重的Chroma文档ID。"""
        if not document.id:
            raise ValueError(
                "混合检索要求每个文档都具有Chroma文档ID"
            )

        return str(document.id)

    def search(
            self,
            query: str,
            k: int,
    ) -> list[HybridSearchResult]:
        """分别执行两路召回，再返回RRF融合结果。"""
        if k <= 0:
            # 在调用远程向量模型前检查参数，避免无效API请求。
            raise ValueError("k必须是大于0的整数")

        vector_matches = (
            self.vector_store_service
            .search_with_relevance_scores(
                query,
                k=self.vector_candidate_k,
            )
        )
        bm25_matches = self.bm25_retriever.search(
            query,
            k=self.bm25_candidate_k,
        )

        return self.fuse(
            vector_matches=vector_matches,
            bm25_matches=bm25_matches,
            k=k,
        )

    def fuse(
            self,
            vector_matches: Sequence[
                tuple[Document, float]
            ],
            bm25_matches: Sequence[
                tuple[Document, float]
            ],
            k: int,
    ) -> list[HybridSearchResult]:
        """融合已经完成召回的两组候选，不再次调用检索服务。"""
        if k <= 0:
            raise ValueError("k必须是大于0的整数")

        candidates: dict[str, _FusionCandidate] = {}

        for rank, (
                document,
                score,
        ) in enumerate(
            vector_matches,
            start=1,
        ):
            document_id = self._get_document_id(
                document
            )
            # dict.setdefault：key不存在就新建_FusionCandidate对象；存在就直接拿旧对象。
            candidate = candidates.setdefault(
                document_id,
                _FusionCandidate(document=document),
            )

            # 如果这个候选还没有记录向量侧信息
            # 同一路检索若意外返回重复ID，只使用首次出现的最高排名。
            if candidate.vector_rank is None:
                candidate.vector_rank = rank
                candidate.vector_score = float(score)
                candidate.fusion_score += (
                        1
                        / (
                                self.rrf_constant
                                + rank
                        )
                )

        for rank, (
                document,
                score,
        ) in enumerate(
            bm25_matches,
            start=1,
        ):
            document_id = self._get_document_id(
                document
            )
            candidate = candidates.setdefault(
                document_id,
                _FusionCandidate(document=document),
            )

            if candidate.bm25_rank is None:
                candidate.bm25_rank = rank
                candidate.bm25_score = float(score)
                candidate.fusion_score += (
                        1
                        / (
                                self.rrf_constant
                                + rank
                        )
                )

        # RRF分数越高越靠前；同分时按文档ID排序，保证结果可重复。
        ranked_candidates = sorted(
            candidates.items(),
            key=lambda item: (
                -item[1].fusion_score,
                item[0],
            ),
        )

        return [
            HybridSearchResult(
                document=candidate.document,
                fusion_score=candidate.fusion_score,
                vector_rank=candidate.vector_rank,
                bm25_rank=candidate.bm25_rank,
                vector_score=candidate.vector_score,
                bm25_score=candidate.bm25_score,
            )
            for _, candidate in ranked_candidates[:k]
        ]
