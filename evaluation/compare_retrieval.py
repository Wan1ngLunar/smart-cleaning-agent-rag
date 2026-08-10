from dataclasses import dataclass

from evaluation.evaluate_retrieval import (
    TOP_K,
    RetrievalCase,
    get_source_filename,
    load_cases,
)
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf

# 从统一配置读取A/B实验选出的最优参数，避免评测与生产配置不一致。
HYBRID_RETRIEVAL_CONF = chroma_conf[
    "hybrid_retrieval"
]
VECTOR_CANDIDATE_K = int(
    HYBRID_RETRIEVAL_CONF[
        "vector_candidate_k"
    ]
)
BM25_CANDIDATE_K = int(
    HYBRID_RETRIEVAL_CONF[
        "bm25_candidate_k"
    ]
)
RRF_CONSTANT = int(
    HYBRID_RETRIEVAL_CONF[
        "rrf_constant"
    ]
)


@dataclass(frozen=True)
class RankingMetrics:
    """保存只针对正例计算的检索排名指标。"""

    hit_at_1: float     # Hit@1：正确文档排在第1位的占比
    hit_at_k: float     # Hit@K：正确文档出现在前K条的占比
    mrr: float          # MRR 倒数平均排名，综合指标，越高越好
    positive_count: int # 有效正例数量


@dataclass(frozen=True)
class RankChange:
    """保存一条正例在两种检索方式下的排名变化。"""

    case_id: str
    vector_rank: int | None
    hybrid_rank: int | None


@dataclass(frozen=True)
class RankingChanges:
    """分类保存混合检索带来的提升、退化和不变数量。"""

    improved: tuple[RankChange, ...]   # 混合检索排名变好了
    regressed: tuple[RankChange, ...]  # 混合检索反而变差
    unchanged_count: int               # 排名没有变化


def find_expected_source_rank(
    case: RetrievalCase,
    sources: tuple[str, ...],
) -> int | None:
    """查找case中预期来源在sources首次出现的位置，排名从1开始。"""
    for rank, source in enumerate(
        sources,
        start=1,
    ):
        if source in case.expected_sources:
            return rank

    return None


def calculate_ranking_metrics(
    cases: tuple[RetrievalCase, ...],
    sources_by_case: dict[str, tuple[str, ...]], # 字典 {case_id: (来源文件名1,来源文件名2,...)}，每个问题检索返回的文档来源文件名
    k: int = TOP_K,
) -> RankingMetrics:
    """根据正例来源排名计算Hit@1、Hit@K和MRR。"""
    if k <= 0:
        raise ValueError("k必须是大于0的整数")

    positive_cases = tuple(
        case
        for case in cases
        if case.kind == "positive"
    )

    if not positive_cases:
        raise ValueError(
            "至少需要一条正例才能计算排名指标"
        )

    ranks: list[int | None] = []

    for case in positive_cases:
        if case.case_id not in sources_by_case:
            raise ValueError(
                f"缺少用例{case.case_id}的检索结果"
            )

        ranks.append(
            find_expected_source_rank(
                case,
                sources_by_case[case.case_id],
            )
        )

    hit_at_1 = sum(
        rank == 1
        for rank in ranks
    ) / len(ranks)

    hit_at_k = sum(
        rank is not None and rank <= k
        for rank in ranks
    ) / len(ranks)

    mrr = sum(
        1 / rank if rank is not None else 0.0
        for rank in ranks
    ) / len(ranks)

    return RankingMetrics(
        hit_at_1=hit_at_1,
        hit_at_k=hit_at_k,
        mrr=mrr,
        positive_count=len(positive_cases),
    )


def compare_rankings(
    cases: tuple[RetrievalCase, ...],
    vector_sources_by_case: dict[
        str,
        tuple[str, ...],
    ],
    hybrid_sources_by_case: dict[
        str,
        tuple[str, ...],
    ],
) -> RankingChanges:
    """比较每条正例的向量排名和混合检索排名。"""
    improved: list[RankChange] = []
    regressed: list[RankChange] = []
    unchanged_count = 0

    for case in cases:
        if case.kind != "positive":
            continue

        if (
            case.case_id not in vector_sources_by_case
            or case.case_id not in hybrid_sources_by_case
        ):
            raise ValueError(
                f"缺少用例{case.case_id}的对比结果"
            )

        vector_rank = find_expected_source_rank(
            case,
            vector_sources_by_case[case.case_id],
        )
        hybrid_rank = find_expected_source_rank(
            case,
            hybrid_sources_by_case[case.case_id],
        )

        change = RankChange(
            case_id=case.case_id,
            vector_rank=vector_rank,
            hybrid_rank=hybrid_rank,
        )

        # None表示未进入Top-K，因此比较时视为无穷大。
        vector_value = (
            float("inf")
            if vector_rank is None
            else vector_rank
        )
        hybrid_value = (
            float("inf")
            if hybrid_rank is None
            else hybrid_rank
        )

        if hybrid_value < vector_value:
            improved.append(change)
        elif hybrid_value > vector_value:
            regressed.append(change)
        else:
            unchanged_count += 1

    return RankingChanges(
        improved=tuple(improved),
        regressed=tuple(regressed),
        unchanged_count=unchanged_count,
    )


def format_rank(rank: int | None) -> str:
    """将未命中的None转换成便于阅读的中文。"""
    return "未命中" if rank is None else str(rank)


if __name__ == "__main__":
    loaded_cases = load_cases()
    positive_cases = tuple(
        case
        for case in loaded_cases
        if case.kind == "positive"
    )

    print("检索A/B评测初始化")
    print("完整用例数量：", len(loaded_cases))
    print("本轮正例数量：", len(positive_cases))
    print(
        "说明：负例用于端到端拒答评测，"
        "不参与本轮排名指标。"
    )

    # 只创建一次Chroma连接和一次内存BM25索引。
    vector_store_service = VectorStoreService()
    documents = (
        vector_store_service.get_all_documents()
    )
    bm25_retriever = BM25Retriever(documents)
    hybrid_retriever = HybridRetriever(
        vector_store_service=vector_store_service,
        bm25_retriever=bm25_retriever,
        vector_candidate_k=VECTOR_CANDIDATE_K,
        bm25_candidate_k=BM25_CANDIDATE_K,
        rrf_constant=RRF_CONSTANT,
    )

    vector_sources_by_case: dict[
        str,
        tuple[str, ...],
    ] = {}
    hybrid_sources_by_case: dict[
        str,
        tuple[str, ...],
    ] = {}

    for index, case in enumerate(
        positive_cases,
        start=1,
    ):
        print(
            f"[{index}/{len(positive_cases)}] "
            f"正在评测：{case.case_id}"
        )

        # 每条用例只执行一次远程向量检索。
        vector_matches = (
            vector_store_service
            .search_with_relevance_scores(
                case.query,
                k=VECTOR_CANDIDATE_K,
            )
        )
        bm25_matches = bm25_retriever.search(
            case.query,
            k=BM25_CANDIDATE_K,
        )

        # 纯向量基线只取前3名，与原有评测保持一致。
        vector_sources_by_case[case.case_id] = tuple(
            get_source_filename(document)
            for document, _ in vector_matches[:TOP_K]
        )

        # 复用上面的两路候选，不再重复调用向量模型。
        hybrid_results = hybrid_retriever.fuse(
            vector_matches=vector_matches,
            bm25_matches=bm25_matches,
            k=TOP_K,
        )
        hybrid_sources_by_case[case.case_id] = tuple(
            get_source_filename(result.document)
            for result in hybrid_results
        )

    vector_metrics = calculate_ranking_metrics(
        loaded_cases,
        vector_sources_by_case,
    )
    hybrid_metrics = calculate_ranking_metrics(
        loaded_cases,
        hybrid_sources_by_case,
    )
    changes = compare_rankings(
        loaded_cases,
        vector_sources_by_case,
        hybrid_sources_by_case,
    )

    print()
    print("纯向量检索基线")
    print(f"Hit@1：{vector_metrics.hit_at_1:.2%}")
    print(f"Hit@{TOP_K}：{vector_metrics.hit_at_k:.2%}")
    print(f"MRR：{vector_metrics.mrr:.4f}")

    print()
    print("RRF混合检索")
    print(f"Hit@1：{hybrid_metrics.hit_at_1:.2%}")
    print(f"Hit@{TOP_K}：{hybrid_metrics.hit_at_k:.2%}")
    print(f"MRR：{hybrid_metrics.mrr:.4f}")

    print()
    print("指标变化")
    print(
        "Hit@1变化："
        f"{hybrid_metrics.hit_at_1 - vector_metrics.hit_at_1:+.2%}"
    )
    print(
        f"Hit@{TOP_K}变化："
        f"{hybrid_metrics.hit_at_k - vector_metrics.hit_at_k:+.2%}"
    )
    print(
        "MRR变化："
        f"{hybrid_metrics.mrr - vector_metrics.mrr:+.4f}"
    )

    print()
    print("排名提升用例")
    if not changes.improved:
        print("无")
    else:
        for change in changes.improved:
            print(
                f"{change.case_id} | "
                f"向量：{format_rank(change.vector_rank)} | "
                f"混合：{format_rank(change.hybrid_rank)}"
            )

    print()
    print("排名退化用例")
    if not changes.regressed:
        print("无")
    else:
        for change in changes.regressed:
            print(
                f"{change.case_id} | "
                f"向量：{format_rank(change.vector_rank)} | "
                f"混合：{format_rank(change.hybrid_rank)}"
            )

    print()
    print("排名不变用例数量：", changes.unchanged_count)
