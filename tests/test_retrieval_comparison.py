import pytest

from evaluation.compare_retrieval import (
    calculate_ranking_metrics,
    compare_rankings,
)
from evaluation.evaluate_retrieval import RetrievalCase


def make_case(
    case_id: str,
    kind: str,
    expected_sources: tuple[str, ...],
) -> RetrievalCase:
    """创建不依赖真实YAML文件的评测用例。"""
    return RetrievalCase(
        case_id=case_id,
        kind=kind,
        query=f"{case_id}的测试问题",
        expected_sources=expected_sources,
    )


def test_calculate_ranking_metrics_uses_positive_cases_only():
    """排名指标应忽略负例，并正确计算Hit和MRR。"""
    cases = (
        make_case(
            "top_1",
            "positive",
            ("正确来源A.txt",),
        ),
        make_case(
            "top_2",
            "positive",
            ("正确来源B.txt",),
        ),
        make_case(
            "negative",
            "negative",
            (),
        ),
    )
    sources_by_case = {
        "top_1": (
            "正确来源A.txt",
            "其他来源.txt",
        ),
        "top_2": (
            "其他来源.txt",
            "正确来源B.txt",
        ),
    }

    metrics = calculate_ranking_metrics(
        cases,
        sources_by_case,
        k=3,
    )

    assert metrics.positive_count == 2
    assert metrics.hit_at_1 == 0.5
    assert metrics.hit_at_k == 1.0
    assert metrics.mrr == 0.75


def test_compare_rankings_classifies_changes():
    """排名前移、后退和不变应分别归类。"""
    cases = (
        make_case(
            "improved",
            "positive",
            ("来源A.txt",),
        ),
        make_case(
            "regressed",
            "positive",
            ("来源B.txt",),
        ),
        make_case(
            "unchanged",
            "positive",
            ("来源C.txt",),
        ),
    )

    vector_sources = {
        "improved": ("其他来源.txt",),
        "regressed": ("来源B.txt",),
        "unchanged": (
            "其他来源.txt",
            "来源C.txt",
        ),
    }
    hybrid_sources = {
        "improved": ("来源A.txt",),
        "regressed": (
            "其他来源1.txt",
            "其他来源2.txt",
            "来源B.txt",
        ),
        "unchanged": (
            "其他来源.txt",
            "来源C.txt",
        ),
    }

    changes = compare_rankings(
        cases,
        vector_sources,
        hybrid_sources,
    )

    assert [
        change.case_id
        for change in changes.improved
    ] == ["improved"]
    assert [
        change.case_id
        for change in changes.regressed
    ] == ["regressed"]
    assert changes.unchanged_count == 1


def test_calculate_ranking_metrics_rejects_missing_results():
    """正例结果缺失说明评测流程不完整，应明确报错。"""
    cases = (
        make_case(
            "missing",
            "positive",
            ("正确来源.txt",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="缺少用例missing的检索结果",
    ):
        calculate_ranking_metrics(
            cases,
            {},
        )
