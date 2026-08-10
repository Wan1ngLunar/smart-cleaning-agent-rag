from dataclasses import dataclass
from itertools import product

from evaluation.compare_retrieval import (
    calculate_ranking_metrics,
    compare_rankings,
)
from evaluation.evaluate_retrieval import (
    TOP_K,
    get_source_filename,
    load_cases,
)
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import VectorStoreService

"""
RRF 混合检索参数扫描脚本（网格搜索）
目的：自动遍历多组超参数，找到 vector_candidate_k、bm25_candidate_k、rrf_constant 最优组合。
关键点：只调用一次向量库、BM25，把候选全部缓存下来，后面所有参数组合只做内存切片 + fuse 融合，不再访问 Chroma / 大模型，速度飞快。
"""
# 网格搜索范围，共2 × 3 × 5 = 30组参数。
VECTOR_CANDIDATE_VALUES = (10, 20)
BM25_CANDIDATE_VALUES = (5, 10, 20)
RRF_CONSTANT_VALUES = (1, 5, 10, 30, 60)


@dataclass(frozen=True)
class ParameterEvaluation:
    """保存一组混合检索参数及其评测结果。"""

    vector_candidate_k: int
    bm25_candidate_k: int
    rrf_constant: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    improved_count: int
    regressed_count: int


def rank_parameter_results(
    results: list[ParameterEvaluation],
) -> tuple[ParameterEvaluation, ...]:
    """按照检索效果、退化数量和运行成本排列参数。"""
    return tuple(
        sorted(
            results,
            key=lambda result: (
                # 优先选择Hit@1、Hit@3和MRR更高的参数。
                -result.hit_at_1,
                -result.hit_at_k,
                -result.mrr,
                # 指标相同时，优先选择退化更少、提升更多的参数。
                result.regressed_count,
                -result.improved_count,
                # 效果相同时减少向量和BM25候选数量。
                result.vector_candidate_k,
                result.bm25_candidate_k,
                # 其余条件相同时，选择更温和的较大RRF常数。
                -result.rrf_constant,
            ),
        )
    )


def print_parameter_result(
    prefix: str,
    result: ParameterEvaluation,
) -> None:
    """使用统一格式打印一组参数结果。"""
    print(
        f"{prefix} | "
        f"向量候选：{result.vector_candidate_k:2d} | "
        f"BM25候选：{result.bm25_candidate_k:2d} | "
        f"RRF常数：{result.rrf_constant:2d} | "
        f"Hit@1：{result.hit_at_1:.2%} | "
        f"Hit@{TOP_K}：{result.hit_at_k:.2%} | "
        f"MRR：{result.mrr:.4f} | "
        f"提升：{result.improved_count} | "
        f"退化：{result.regressed_count}"
    )


if __name__ == "__main__":
    loaded_cases = load_cases()
    positive_cases = tuple(
        case
        for case in loaded_cases
        if case.kind == "positive"
    )

    vector_store_service = VectorStoreService()
    documents = (
        vector_store_service.get_all_documents()
    )
    bm25_retriever = BM25Retriever(documents)

    # 只按参数范围中的最大值执行一次召回，后续组合复用缓存。
    maximum_vector_k = max(
        VECTOR_CANDIDATE_VALUES
    )
    maximum_bm25_k = max(
        BM25_CANDIDATE_VALUES
    )

    vector_matches_by_case = {}
    bm25_matches_by_case = {}
    vector_sources_by_case = {}

    for index, case in enumerate(
        positive_cases,
        start=1,
    ):
        print(
            f"[{index}/{len(positive_cases)}] "
            f"正在缓存候选：{case.case_id}"
        )

        vector_matches = (
            vector_store_service
            .search_with_relevance_scores(
                case.query,
                k=maximum_vector_k,
            )
        )
        bm25_matches = bm25_retriever.search(
            case.query,
            k=maximum_bm25_k,
        )

        vector_matches_by_case[
            case.case_id
        ] = vector_matches
        bm25_matches_by_case[
            case.case_id
        ] = bm25_matches
        vector_sources_by_case[
            case.case_id
        ] = tuple(
            get_source_filename(document)
            for document, _ in vector_matches[:TOP_K]
        )

    vector_metrics = calculate_ranking_metrics(
        loaded_cases,
        vector_sources_by_case,
    )

    parameter_results: list[
        ParameterEvaluation
    ] = []

    for (
        vector_candidate_k,
        bm25_candidate_k,
        rrf_constant,
    ) in product(
        VECTOR_CANDIDATE_VALUES,
        BM25_CANDIDATE_VALUES,
        RRF_CONSTANT_VALUES,
    ):
        hybrid_retriever = HybridRetriever(
            vector_store_service=vector_store_service,
            bm25_retriever=bm25_retriever,
            vector_candidate_k=vector_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            rrf_constant=rrf_constant,
        )
        hybrid_sources_by_case = {}

        for case in positive_cases:
            hybrid_results = hybrid_retriever.fuse(
                vector_matches=(
                    vector_matches_by_case[
                        case.case_id
                    ][:vector_candidate_k]
                ),
                bm25_matches=(
                    bm25_matches_by_case[
                        case.case_id
                    ][:bm25_candidate_k]
                ),
                k=TOP_K,
            )
            hybrid_sources_by_case[
                case.case_id
            ] = tuple(
                get_source_filename(
                    result.document
                )
                for result in hybrid_results
            )

        metrics = calculate_ranking_metrics(
            loaded_cases,
            hybrid_sources_by_case,
        )
        changes = compare_rankings(
            loaded_cases,
            vector_sources_by_case,
            hybrid_sources_by_case,
        )

        parameter_results.append(
            ParameterEvaluation(
                vector_candidate_k=(
                    vector_candidate_k
                ),
                bm25_candidate_k=(
                    bm25_candidate_k
                ),
                rrf_constant=rrf_constant,
                hit_at_1=metrics.hit_at_1,
                hit_at_k=metrics.hit_at_k,
                mrr=metrics.mrr,
                improved_count=len(
                    changes.improved
                ),
                regressed_count=len(
                    changes.regressed
                ),
            )
        )

    ranked_results = rank_parameter_results(
        parameter_results
    )

    print()
    print("纯向量基线")
    print(f"Hit@1：{vector_metrics.hit_at_1:.2%}")
    print(f"Hit@{TOP_K}：{vector_metrics.hit_at_k:.2%}")
    print(f"MRR：{vector_metrics.mrr:.4f}")

    print()
    print("全部参数组合")
    for result in parameter_results:
        print_parameter_result(
            "参数",
            result,
        )

    print()
    print("综合排名最高的10组参数")
    for rank, result in enumerate(
        ranked_results[:10],
        start=1,
    ):
        print_parameter_result(
            f"第{rank}名",
            result,
        )

    recommended = ranked_results[0]

    print()
    print("推荐配置")
    print(
        "vector_candidate_k：",
        recommended.vector_candidate_k,
    )
    print(
        "bm25_candidate_k：",
        recommended.bm25_candidate_k,
    )
    print(
        "rrf_constant：",
        recommended.rrf_constant,
    )
