from langchain_core.documents import Document

from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import QwenReranker, RerankerError, build_reranker
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf, rag_conf
from utils.logger_handler import logger


# 完整的数据流是：
# 用户问题
#   → 向量召回10条
#   → BM25召回20条
#   → RRF融合为10条
#   → qwen3-rerank重排
#   → 返回Top-3
class HybridRerankRetriever:
    """执行向量召回、BM25召回、RRF融合和模型重排序。"""

    def __init__(
        self,
        vector_store_service: VectorStoreService | None = None,
        reranker: QwenReranker | None = None,
    ):
        """初始化检索组件，并读取已经通过评测选出的参数。"""
        hybrid_config = chroma_conf["hybrid_retrieval"]
        rerank_config = rag_conf["rerank"]

        # 测试时允许注入假的向量库，避免访问真实Chroma和Embedding接口。
        self.vector_store_service = (
            vector_store_service or VectorStoreService()
        )

        # BM25需要在初始化时读取知识库文档并建立倒排统计。
        all_documents = self.vector_store_service.get_all_documents()
        self.bm25_retriever = BM25Retriever(all_documents)

        # RRF负责合并向量检索排名和BM25检索排名。
        self.hybrid_retriever = HybridRetriever(
            vector_store_service=self.vector_store_service,
            bm25_retriever=self.bm25_retriever,
            vector_candidate_k=int(
                hybrid_config["vector_candidate_k"]
            ),
            bm25_candidate_k=int(
                hybrid_config["bm25_candidate_k"]
            ),
            rrf_constant=int(
                hybrid_config["rrf_constant"]
            ),
        )

        # 测试时允许注入假的重排序器，正式运行时创建qwen3-rerank客户端。
        self.reranker = reranker or build_reranker()

        # 这些参数来自前面完成的网格搜索和重排序实验。
        self.vector_candidate_k = int(
            hybrid_config["vector_candidate_k"]
        )
        self.bm25_candidate_k = int(
            hybrid_config["bm25_candidate_k"]
        )
        self.rerank_candidate_k = int(
            rerank_config["candidate_k"]
        )
        self.top_n = int(rerank_config["top_n"])

        # 只过滤明显无关的向量候选；BM25候选不受该阈值限制。
        self.min_vector_relevance_score = float(
            rag_conf["min_relevance_score"]
        )

        self._closed = False

    def retrieve(self, query: str) -> list[Document]:
        """返回经过混合召回和重排序后的Top-N文档。"""
        if self._closed:
            raise RuntimeError("检索流水线已经关闭，不能继续执行检索")

        normalized_query = query.strip()
        if not normalized_query:
            return []

        # 第一条召回路线：语义向量检索。
        vector_matches = (
            self.vector_store_service.search_with_relevance_scores(
                normalized_query,
                k=self.vector_candidate_k,
            )
        )

        # 过滤空片段和明显低于阈值的向量结果。
        filtered_vector_matches = [
            (document, score)
            for document, score in vector_matches
            if document.page_content.strip()
            and score >= self.min_vector_relevance_score
        ]

        # 第二条召回路线：BM25关键词检索。
        bm25_matches = self.bm25_retriever.search(
            normalized_query,
            k=self.bm25_candidate_k,
        )

        # 使用RRF合并两条召回路线，保留10个候选供重排序。
        hybrid_candidates = self.hybrid_retriever.fuse(
            vector_matches=filtered_vector_matches,
            bm25_matches=bm25_matches,
            k=self.rerank_candidate_k,
        )

        if not hybrid_candidates:
            return []

        candidate_documents = [
            candidate.document
            for candidate in hybrid_candidates
            if candidate.document.page_content.strip()
        ]

        if not candidate_documents:
            return []

        try:
            # qwen3-rerank根据“问题是否能由片段回答”重新排列候选。
            rerank_results = self.reranker.rerank(
                query=normalized_query,
                documents=candidate_documents,
                top_n=self.top_n,
            )
        except RerankerError as error:
            # 外部重排序服务不可用时降级到RRF结果，避免整个RAG服务中断。
            logger.warning(
                "[HybridRerankRetriever]重排序失败，"
                "已降级为RRF混合检索结果：%s",
                error,
            )
            return candidate_documents[: self.top_n]

        return [
            result.document
            for result in rerank_results
            if result.document.page_content.strip()
        ]

    def close(self) -> None:
        """关闭重排序器持有的HTTP连接。"""
        if self._closed:
            return

        self.reranker.close()
        self._closed = True
