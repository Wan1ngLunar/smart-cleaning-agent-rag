from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from evaluation.compare_retrieval import format_rank
from evaluation.evaluate_retrieval import (
    get_source_filename,
    load_cases,
)
from model.factory import chat_model
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import build_reranker
from rag.vector_store import VectorStoreService
from utils.config_handler import (
    chroma_conf,
    rag_conf,
)

# 查询改写（Query Rewrite）A/B 评测脚本。
# 业务背景：用户原始问句口语化、简短，向量 / BM25 检索效果差；想用大模型把用户问题改写成更适合检索的句子。
# 实验设计：同一测试用例跑两组对比
# A 组：直接用用户原始 query做召回
# B 组：用大模型改写后的 query做召回，但是重排序依旧使用原始用户问题
# 最后对比 Hit@1、MRR，看改写之后检索效果有没有变好，判断要不要把查询改写逻辑上线到生产。

# 只评估重排序后正确来源仍未进入Top-1的固定用例。
TARGET_CASE_IDS = (
    "carpet_dust_boost",
    "long_charge_battery_protection",
    "forbidden_zone_still_entered",
    "remote_control_no_response",
)

QUERY_REWRITE_PROMPT = (
    "你是扫地机器人本地知识库的检索查询改写器。"
    "请把用户问题改写成一条更适合向量检索和关键词检索的中文查询。"
    "必须保留原问题中的设备对象、故障现象、数值、条件和提问目标。"
    "禁止回答问题，禁止补充原问题没有提供的事实，禁止改变问题范围。"
    "只输出改写后的查询，不要添加标题、解释、引号或编号。"
)


def rewrite_query(query: str) -> str:
    """调用聊天模型生成一条保持原意的检索查询。"""
    response = chat_model.invoke(
        [
            SystemMessage(
                content=QUERY_REWRITE_PROMPT,
            ),
            HumanMessage(content=query),
        ]
    )

    if not isinstance(response.content, str):
        raise TypeError(
            "查询改写模型没有返回纯文本"
        )

    # 合并模型可能产生的换行和多余空格，得到单行检索查询。
    rewritten_query = " ".join(
        response.content.split()
    )

    if not rewritten_query:
        raise ValueError(
            "查询改写模型返回了空文本"
        )

    return rewritten_query


def find_expected_rank(
    sources: tuple[str, ...], # 重排之后返回的文档文件名列表，顺序就是检索排名
    expected_sources: tuple[str, ...],
) -> int | None:
    """返回第一个正确来源的排名，未进入Top-3时返回None。"""
    expected_source_set = set(expected_sources)

    for rank, source in enumerate(
        sources,
        start=1,
    ):
        if source in expected_source_set:
            return rank

    return None


def calculate_mrr(
    ranks: list[int | None],
) -> float:
    """计算当前4条用例的平均倒数排名。"""
    return sum(
        0.0 if rank is None else 1.0 / rank
        for rank in ranks
    ) / len(ranks)


def retrieve_sources(
    retrieval_query: str,
    original_query: str,
    vector_store_service: VectorStoreService,
    bm25_retriever: BM25Retriever,
    hybrid_retriever: HybridRetriever,
    reranker,
) -> tuple[str, ...]:
    """执行与生产环境一致的召回、融合和重排序流程。"""
    hybrid_config = chroma_conf[
        "hybrid_retrieval"
    ]
    rerank_config = rag_conf["rerank"]

    vector_matches = (
        vector_store_service
        .search_with_relevance_scores(
            retrieval_query,
            k=int(
                hybrid_config[
                    "vector_candidate_k"
                ]
            ),
        )
    )

    min_relevance_score = float(
        rag_conf["min_relevance_score"]
    )

    # 与生产流水线一致，只过滤向量路线中的明显低分片段。
    filtered_vector_matches = [
        (document, score)
        for document, score in vector_matches
        if document.page_content.strip()
        and score >= min_relevance_score
    ]

    bm25_matches = bm25_retriever.search(
        retrieval_query,
        k=int(
            hybrid_config["bm25_candidate_k"]
        ),
    )

    hybrid_candidates = hybrid_retriever.fuse(
        vector_matches=filtered_vector_matches,
        bm25_matches=bm25_matches,
        k=int(rerank_config["candidate_k"]),
    )

    if not hybrid_candidates:
        return ()

    # 用改写后的句子去向量库、BM25 拿候选文档（扩大拿到正确文档的概率）
    # 拿到一堆候选片段之后，重排序的时候，喂给重排模型的依旧是用户原汁原味的提问
    # 为什么要这么做？
    # 如果改写后的句子和用户真实意图有微小偏差，重排用改写句，会把文档排错顺序。
    # 召回阶段只负责 “把正确文档捞出来”；重排阶段依旧尊重用户真实提问。
    reranked_results = reranker.rerank(
        query=original_query,
        documents=[
            result.document
            for result in hybrid_candidates
        ],
        top_n=int(rerank_config["top_n"]),
    )

    return tuple(
        get_source_filename(result.document)
        for result in reranked_results
    )


if __name__ == "__main__":
    loaded_cases = load_cases()
    cases_by_id = {
        case.case_id: case
        for case in loaded_cases
    }

    missing_case_ids = [
        case_id
        for case_id in TARGET_CASE_IDS
        if case_id not in cases_by_id
    ]

    if missing_case_ids:
        raise ValueError(
            "评测集缺少查询改写用例："
            + ", ".join(missing_case_ids)
        )

    hybrid_config = chroma_conf[
        "hybrid_retrieval"
    ]

    # 整个实验复用同一个Chroma连接、BM25索引和重排序连接池。
    vector_store_service = VectorStoreService()
    all_documents = (
        vector_store_service.get_all_documents()
    )
    bm25_retriever = BM25Retriever(
        all_documents
    )
    hybrid_retriever = HybridRetriever(
        vector_store_service=vector_store_service,
        bm25_retriever=bm25_retriever,
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

    original_ranks: list[int | None] = []
    rewritten_ranks: list[int | None] = []
    improved_count = 0
    regressed_count = 0
    unchanged_count = 0

    with build_reranker() as reranker:
        for case_id in TARGET_CASE_IDS:
            case = cases_by_id[case_id]
            rewritten_query = rewrite_query(
                case.query
            )

            # A组：原问题负责召回和重排序。
            original_sources = retrieve_sources(
                retrieval_query=case.query,
                original_query=case.query,
                vector_store_service=(
                    vector_store_service
                ),
                bm25_retriever=bm25_retriever,
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
            )

            # B组：改写问题负责召回，原问题仍负责最终重排序。
            rewritten_sources = retrieve_sources(
                retrieval_query=rewritten_query,
                original_query=case.query,
                vector_store_service=(
                    vector_store_service
                ),
                bm25_retriever=bm25_retriever,
                hybrid_retriever=hybrid_retriever,
                reranker=reranker,
            )

            original_rank = find_expected_rank(
                original_sources,
                case.expected_sources,
            )
            rewritten_rank = find_expected_rank(
                rewritten_sources,
                case.expected_sources,
            )

            original_ranks.append(original_rank)
            rewritten_ranks.append(rewritten_rank)

            # None表示正确来源没有进入Top-3，可视为比任何有效排名都差。
            original_value = (
                original_rank
                if original_rank is not None
                else 4
            )
            rewritten_value = (
                rewritten_rank
                if rewritten_rank is not None
                else 4
            )

            if rewritten_value < original_value:
                improved_count += 1
                result = "提升"
            elif rewritten_value > original_value:
                regressed_count += 1
                result = "退化"
            else:
                unchanged_count += 1
                result = "不变"

            print("=" * 80)
            print("用例：", case.case_id)
            print("原问题：", case.query)
            print("改写后：", rewritten_query)
            print(
                "原查询正确来源排名：",
                format_rank(original_rank),
            )
            print(
                "改写查询正确来源排名：",
                format_rank(rewritten_rank),
            )
            print(
                "原查询Top-3：",
                ", ".join(original_sources),
            )
            print(
                "改写查询Top-3：",
                ", ".join(rewritten_sources),
            )
            print("变化：", result)

    original_hit_at_1 = sum(
        rank == 1
        for rank in original_ranks
    )
    rewritten_hit_at_1 = sum(
        rank == 1
        for rank in rewritten_ranks
    )

    print()
    print("查询改写定向A/B实验")
    print("用例数量：", len(TARGET_CASE_IDS))
    print(
        "原查询Hit@1："
        f"{original_hit_at_1}/{len(TARGET_CASE_IDS)}"
    )
    print(
        "改写查询Hit@1："
        f"{rewritten_hit_at_1}/{len(TARGET_CASE_IDS)}"
    )
    print(
        "原查询MRR："
        f"{calculate_mrr(original_ranks):.4f}"
    )
    print(
        "改写查询MRR："
        f"{calculate_mrr(rewritten_ranks):.4f}"
    )
    print("提升用例：", improved_count)
    print("退化用例：", regressed_count)
    print("不变用例：", unchanged_count)

    if improved_count > regressed_count:
        print(
            "初步结论：存在净提升，"
            "下一步应扩展到全部36条正例验证。"
        )
    else:
        print(
            "初步结论：没有稳定净提升，"
            "暂不接入生产检索链路。"
        )
