import pytest

from evaluation.evaluate_answerability import (
    SMOKE_CASE_IDS,
    is_case_passed,
    select_cases,
)
from evaluation.evaluate_retrieval import (
    RetrievalCase,
    RetrievalResult,
    calculate_summary,
    load_cases,
)
from rag.rag_service import (
    EMPTY_ANSWER_RESPONSE,
    NO_CONTEXT_RESPONSE,
)


def make_case(
    case_id: str,
    kind: str,
    expected_sources: tuple[str, ...],
) -> RetrievalCase:
    """创建不依赖YAML文件的测试用评估用例。"""
    return RetrievalCase(
        case_id=case_id,
        kind=kind,
        query=f"{case_id}的测试问题",
        expected_sources=expected_sources,
    )


def make_result(
    rank: int,
    source: str,
    score: float,
) -> RetrievalResult:
    """创建不依赖真实Chroma检索的测试结果。"""
    return RetrievalResult(
        rank=rank,
        source=source,
        score=score,
        preview="测试片段预览",
    )


def test_retrieval_cases_file_contains_current_baseline():
    """YAML应至少保留当前建立的正例、负例和冒烟用例。"""
    cases = load_cases()
    positive_count = sum(
        case.kind == "positive"
        for case in cases
    )
    negative_count = sum(
        case.kind == "negative"
        for case in cases
    )
    case_ids = {
        case.case_id
        for case in cases
    }

    # 使用大于等于，允许未来继续增加评估用例。
    assert len(cases) >= 14
    assert positive_count >= 6
    assert negative_count >= 8
    assert set(SMOKE_CASE_IDS).issubset(case_ids)


def test_calculate_summary_reports_score_overlap():
    """负例最高分超过正例最低分时，不应生成候选阈值。"""
    positive_case = make_case(
        "positive_case",
        "positive",
        ("正确来源.txt",),
    )
    negative_case = make_case(
        "negative_case",
        "negative",
        (),
    )

    results_by_case = {
        "positive_case": (
            # 正确来源排在第二名，用于同时验证Hit@1、Hit@3和MRR。
            make_result(1, "错误来源.txt", 0.80),
            make_result(2, "正确来源.txt", 0.70),
        ),
        "negative_case": (
            make_result(1, "相似但无关.txt", 0.90),
        ),
    }

    summary = calculate_summary(
        (positive_case, negative_case),
        results_by_case,
        k=3,
    )

    assert summary.hit_at_1 == 0.0
    assert summary.hit_at_k == 1.0
    assert summary.mrr == 0.5
    assert summary.positive_min_top_1_score == 0.80
    assert summary.negative_max_top_1_score == 0.90
    assert summary.score_gap == pytest.approx(-0.10)
    assert summary.has_score_overlap is True
    assert summary.threshold_candidate is None


def test_calculate_summary_returns_threshold_without_overlap():
    """正例分数全部高于负例时，可以返回两类边界的中点。"""
    positive_case = make_case(
        "positive_case",
        "positive",
        ("正确来源.txt",),
    )
    negative_case = make_case(
        "negative_case",
        "negative",
        (),
    )

    results_by_case = {
        "positive_case": (
            make_result(1, "正确来源.txt", 0.80),
        ),
        "negative_case": (
            make_result(1, "无关来源.txt", 0.20),
        ),
    }

    summary = calculate_summary(
        (positive_case, negative_case),
        results_by_case,
    )

    assert summary.hit_at_1 == 1.0
    assert summary.hit_at_k == 1.0
    assert summary.mrr == 1.0
    assert summary.score_gap == pytest.approx(0.60)
    assert summary.has_score_overlap is False
    assert summary.threshold_candidate == pytest.approx(0.50)


def test_select_cases_supports_smoke_and_full_modes():
    """默认模式应选三条冒烟用例，全量模式应保留全部用例。"""
    loaded_cases = load_cases()

    smoke_cases = select_cases(
        loaded_cases,
        run_all=False,
    )
    all_cases = select_cases(
        loaded_cases,
        run_all=True,
    )

    assert tuple(
        case.case_id
        for case in smoke_cases
    ) == SMOKE_CASE_IDS
    assert all_cases == loaded_cases


def test_select_cases_rejects_missing_smoke_case():
    """冒烟用例被误删时，应在调用真实API之前失败。"""
    incomplete_cases = (
        make_case(
            "unrelated_case",
            "negative",
            (),
        ),
    )

    with pytest.raises(
        ValueError,
        match="缺少端到端冒烟用例",
    ):
        select_cases(
            incomplete_cases,
            run_all=False,
        )


def test_is_case_passed_distinguishes_answers_and_rejections():
    """正例必须有有效答案和来源，负例必须返回统一拒答文本。"""
    positive_case = make_case(
        "positive_case",
        "positive",
        ("正确来源.txt",),
    )
    negative_case = make_case(
        "negative_case",
        "negative",
        (),
    )

    valid_answer = (
        "这是基于知识库生成的有效答案。[1]\n\n"
        "参考来源：\n"
        "[1] 正确来源.txt"
    )
    empty_answer = (
        f"{EMPTY_ANSWER_RESPONSE}\n\n"
        "参考来源：\n"
        "[1] 正确来源.txt"
    )

    assert is_case_passed(positive_case, valid_answer)
    assert not is_case_passed(
        positive_case,
        NO_CONTEXT_RESPONSE,
    )
    assert not is_case_passed(
        positive_case,
        empty_answer,
    )

    assert is_case_passed(
        negative_case,
        NO_CONTEXT_RESPONSE,
    )
    assert not is_case_passed(
        negative_case,
        valid_answer,
    )
