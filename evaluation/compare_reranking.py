from evaluation.compare_retrieval import (
    RankingChanges,
    RankingMetrics,
    calculate_ranking_metrics,
    compare_rankings,
    format_rank,
)
from evaluation.evaluate_retrieval import (
    TOP_K,
    get_source_filename,
    load_cases,
)
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import build_reranker
from rag.vector_store import VectorStoreService
from utils.config_handler import (
    chroma_conf,
    rag_conf,
)


def print_metrics(
    title: str,
    metrics: RankingMetrics,
) -> None:
    """使用统一格式打印一组来源级排名指标。"""
    print()
    print(title)
    print(f"Hit@1：{metrics.hit_at_1:.2%}")
    print(f"Hit@{TOP_K}：{metrics.hit_at_k:.2%}")
    print(f"MRR：{metrics.mrr:.4f}")


def print_changes(
    changes: RankingChanges,
) -> None:
    """打印重排序相对混合Top-3的提升和退化用例。"""
    print()
    print("重排序后排名提升用例")

    if not changes.improved:
        print("无")
    else:
        for change in changes.improved:
            print(
                f"{change.case_id} | "
                f"混合：{format_rank(change.vector_rank)} | "
                f"重排：{format_rank(change.hybrid_rank)}"
            )

    print()
    print("重排序后排名退化用例")

    if not changes.regressed:
        print("无")
    else:
        for change in changes.regressed:
            print(
                f"{change.case_id} | "
                f"混合：{format_rank(change.vector_rank)} | "
                f"重排：{format_rank(change.hybrid_rank)}"
            )

    print()
    print(
        "重排序后排名不变用例数量：",
        changes.unchanged_count,
    )


if __name__ == "__main__":
    loaded_cases = load_cases()
    positive_cases = tuple(
        case
        for case in loaded_cases
        if case.kind == "positive"
    )

    hybrid_config = chroma_conf[
        "hybrid_retrieval"
    ]
    rerank_config = rag_conf["rerank"]

    vector_candidate_k = int(
        hybrid_config["vector_candidate_k"]
    )
    bm25_candidate_k = int(
        hybrid_config["bm25_candidate_k"]
    )
    rrf_constant = int(
        hybrid_config["rrf_constant"]
    )
    rerank_candidate_k = int(
        rerank_config["candidate_k"]
    )
    rerank_top_n = int(
        rerank_config["top_n"]
    )

    print("重排序A/B评测初始化")
    print("完整用例数量：", len(loaded_cases))
    print("本轮正例数量：", len(positive_cases))
    print(
        "处理流程：纯向量 → BM25与RRF → "
        "qwen3-rerank"
    )
    print(
        "说明：负例拒答将在生产链路接入后"
        "通过端到端评测单独验证。"
    )

    # 三个阶段复用同一个Chroma连接和BM25内存索引。
    vector_store_service = VectorStoreService()
    documents = (
        vector_store_service.get_all_documents()
    )
    bm25_retriever = BM25Retriever(documents)
    hybrid_retriever = HybridRetriever(
        vector_store_service=vector_store_service,
        bm25_retriever=bm25_retriever,
        vector_candidate_k=vector_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        rrf_constant=rrf_constant,
    )

    vector_sources_by_case: dict[
        str,
        tuple[str, ...],
    ] = {}
    hybrid_top_3_sources_by_case: dict[
        str,
        tuple[str, ...],
    ] = {}
    hybrid_candidate_sources_by_case: dict[
        str,
        tuple[str, ...],
    ] = {} # 送入 rerank 的候选池
    reranked_sources_by_case: dict[
        str,
        tuple[str, ...],
    ] = {} # rerank 重排输出结果

    # 复用一个HTTP连接池，36条用例各调用一次重排序接口。
    with build_reranker() as reranker:
        for index, case in enumerate(
            positive_cases,
            start=1,
        ):
            print(
                f"[{index}/{len(positive_cases)}] "
                f"正在重排序：{case.case_id}"
            )

            # 每个问题只调用一次Embedding。
            vector_matches = (
                vector_store_service
                .search_with_relevance_scores(
                    case.query,
                    k=vector_candidate_k,
                )
            )
            bm25_matches = bm25_retriever.search(
                case.query,
                k=bm25_candidate_k,
            )

            vector_sources_by_case[
                case.case_id
            ] = tuple(
                get_source_filename(document)
                for document, _ in vector_matches[
                    :TOP_K
                ]
            )

            # 保留10条融合候选，先计算混合Top-3，再交给重排序。
            hybrid_candidates = (
                hybrid_retriever.fuse(
                    vector_matches=vector_matches,
                    bm25_matches=bm25_matches,
                    k=rerank_candidate_k,
                )
            )
            hybrid_candidate_sources_by_case[
                case.case_id
            ] = tuple(
                get_source_filename(
                    result.document
                )
                for result in hybrid_candidates
            )
            hybrid_top_3_sources_by_case[
                case.case_id
            ] = (
                hybrid_candidate_sources_by_case[
                    case.case_id
                ][:TOP_K]
            )

            reranked_results = reranker.rerank(
                query=case.query,
                documents=[
                    result.document
                    for result in hybrid_candidates
                ],
                top_n=rerank_top_n,
            )
            reranked_sources_by_case[
                case.case_id
            ] = tuple(
                get_source_filename(
                    result.document
                )
                for result in reranked_results
            )

    vector_metrics = calculate_ranking_metrics(
        loaded_cases,
        vector_sources_by_case,
    )
    hybrid_metrics = calculate_ranking_metrics(
        loaded_cases,
        hybrid_top_3_sources_by_case,
    )
    candidate_metrics = calculate_ranking_metrics(
        loaded_cases,
        hybrid_candidate_sources_by_case,
        k=rerank_candidate_k,
    )
    reranked_metrics = calculate_ranking_metrics(
        loaded_cases,
        reranked_sources_by_case,
    )
    rerank_changes = compare_rankings(
        loaded_cases,
        hybrid_top_3_sources_by_case,
        reranked_sources_by_case,
    )

    print_metrics(
        "纯向量检索基线",
        vector_metrics,
    )
    print_metrics(
        "BM25与RRF混合Top-3",
        hybrid_metrics,
    )

    print()
    print(
        f"混合候选Recall@{rerank_candidate_k}："
        f"{candidate_metrics.hit_at_k:.2%}"
    )

    print_metrics(
        "qwen3-rerank重排序Top-3",
        reranked_metrics,
    )

    print()
    print("重排序相对混合检索的指标变化")
    print(
        "Hit@1变化："
        f"{reranked_metrics.hit_at_1 - hybrid_metrics.hit_at_1:+.2%}"
    )
    print(
        f"Hit@{TOP_K}变化："
        f"{reranked_metrics.hit_at_k - hybrid_metrics.hit_at_k:+.2%}"
    )
    print(
        "MRR变化："
        f"{reranked_metrics.mrr - hybrid_metrics.mrr:+.4f}"
    )

    print_changes(rerank_changes)
