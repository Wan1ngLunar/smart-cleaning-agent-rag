from collections.abc import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.text_tokenizer import tokenize_for_bm25


class BM25Retriever:
    """对一组LangChain文档建立内存BM25关键词索引。"""

    def __init__(
        self,
        documents: Sequence[Document],
    ):
        # 使用元组保存快照，防止外部列表变化导致文档和索引错位。
        # 列表是可变对象，如果外部代码修改原 list，内部索引和文档顺序会错乱；元组不可修改，固定快照，保证索引、文档一一对应。
        self.documents = tuple(documents)

        # 每个知识片段使用同一个确定性分词器处理。
        self._tokenized_documents = tuple(
            tuple(tokenize_for_bm25(document.page_content))
            for document in self.documents
        )

        # 空文档集合或全部为空白时不能建立有效BM25索引。
        if (
            not self.documents
            or not any(self._tokenized_documents)
        ):
            self._index: BM25Okapi | None = None
            return

        # rank-bm25需要可变的二维词元列表，因此在这里转换一次。
        tokenized_corpus = [
            list(tokens)
            for tokens in self._tokenized_documents
        ]
        self._index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """返回按BM25分数从高到低排列的文档。"""
        if k <= 0:
            raise ValueError("k必须是大于0的整数")

        query_tokens = tokenize_for_bm25(query)

        # 没有索引或问题中没有有效词元时，不返回候选文档。
        if self._index is None or not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        # 只保留至少共享一个词元的文档。
        # 不能简单使用score > 0，因为BM25的IDF可能让高频词得到负分。
        matching_results = [
            (
                index,
                float(scores[index]),
            )
            for index, document_tokens in enumerate(
                self._tokenized_documents
            )
            if query_token_set.intersection(document_tokens)
        ]

        # 优先按BM25 分数从高到低排，分数相同时按原始文档顺序排列，使评测结果可以重复。
        matching_results.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        return [
            (
                self.documents[index],
                score,
            )
            for index, score in matching_results[:k]
        ]
