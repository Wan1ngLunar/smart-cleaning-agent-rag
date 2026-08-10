from evaluation.tune_hybrid_retrieval import (
    ParameterEvaluation,
    rank_parameter_results,
)


def make_result(
    vector_candidate_k: int,
    bm25_candidate_k: int,
    rrf_constant: int,
    hit_at_1: float = 0.86,
    hit_at_k: float = 0.97,
    mrr: float = 0.91,
    improved_count: int = 5,
    regressed_count: int = 2,
) -> ParameterEvaluation:
    """创建用于测试参数排序规则的结果。"""
    return ParameterEvaluation(
        vector_candidate_k=vector_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        rrf_constant=rrf_constant,
        hit_at_1=hit_at_1,
        hit_at_k=hit_at_k,
        mrr=mrr,
        improved_count=improved_count,
        regressed_count=regressed_count,
    )


def test_rank_parameter_results_prefers_better_metrics():
    """主要检索指标更高的参数应优先。"""
    better = make_result(
        10,
        20,
        10,
        hit_at_1=0.86,
    )
    worse = make_result(
        10,
        20,
        5,
        hit_at_1=0.83,
    )

    ranked = rank_parameter_results(
        [worse, better]
    )

    assert ranked[0] == better


def test_rank_parameter_results_uses_stable_tie_breakers():
    """指标相同时应减少候选，并选择较温和的RRF常数。"""
    selected = make_result(
        10,
        20,
        10,
    )
    smaller_rrf_constant = make_result(
        10,
        20,
        5,
    )
    more_vector_candidates = make_result(
        20,
        20,
        10,
    )

    ranked = rank_parameter_results(
        [
            smaller_rrf_constant,
            more_vector_candidates,
            selected,
        ]
    )

    assert ranked[0] == selected
