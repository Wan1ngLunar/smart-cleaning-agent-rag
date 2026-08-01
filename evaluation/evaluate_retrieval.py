from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.documents import Document

from rag.vector_store import VectorStoreService

# 使用脚本自身的位置计算用例文件路径，避免运行目录不同导致找不到文件。
CASES_PATH = Path(__file__).resolve().parent / "retrieval_cases.yml"

# 每个问题检查排名最靠前的3个知识片段，与当前RAG配置保持一致。
TOP_K = 3


@dataclass(frozen=True)
class RetrievalCase:
    """保存一条经过校验的检索评估用例。"""

    case_id: str
    kind: str
    query: str
    expected_sources: tuple[str, ...]

@dataclass(frozen=True)
class RetrievalResult:
    """保存一个带排名和相关性分数的检索结果。"""

    rank: int
    source: str
    score: float
    preview: str

@dataclass(frozen=True)
class EvaluationSummary:
    """保存一次完整检索评估的汇总指标。"""

    hit_at_1: float
    hit_at_k: float
    mrr: float
    positive_min_top_1_score: float
    negative_max_top_1_score: float
    # 正数表示正负例之间存在间隔，负数表示分数已经重叠。
    score_gap: float

    # 记录是否存在“负例分数高于正例分数”的情况。
    has_score_overlap: bool

    # 只有分数没有重叠时，才允许生成候选阈值。
    threshold_candidate: float | None

def load_cases(path: Path = CASES_PATH) -> tuple[RetrievalCase, ...]:
    """读取并校验YAML，防止错误用例进入真实评估。"""
    raw_data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_data, dict):
        raise ValueError("评估文件顶层必须是YAML对象")

    raw_cases = raw_data.get("cases")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases必须是非空列表")

    cases: list[RetrievalCase] = [] # 存放最终校验完成的 RetrievalCase 实例
    seen_ids: set[str] = set() # 集合，用来检测case_id重复（不允许两个用例同一个 id）

    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"第{index}条用例必须是YAML对象")

        case_id = str(raw_case.get("id") or "").strip()
        kind = str(raw_case.get("kind") or "").strip()
        query = str(raw_case.get("query") or "").strip()
        raw_sources = raw_case.get("expected_sources")

        if not case_id:
            raise ValueError(f"第{index}条用例缺少id")

        if case_id in seen_ids:
            raise ValueError(f"用例id重复：{case_id}")

        if kind not in {"positive", "negative"}:
            raise ValueError(
                f"用例{case_id}的kind必须是positive或negative"
            )

        if not query:
            raise ValueError(f"用例{case_id}缺少query")

        if not isinstance(raw_sources, list) or not all(
            isinstance(source, str) and source.strip()
            for source in raw_sources
        ):
            raise ValueError(
                f"用例{case_id}的expected_sources必须是字符串列表"
            )

        expected_sources = tuple(
            source.strip()
            for source in raw_sources
        )

        if kind == "positive" and not expected_sources:
            raise ValueError(
                f"正例{case_id}必须至少提供一个预期来源"
            )

        if kind == "negative" and expected_sources:
            raise ValueError(
                f"负例{case_id}不应提供预期来源"
            )

        seen_ids.add(case_id)
        cases.append(
            RetrievalCase(
                case_id=case_id,
                kind=kind,
                query=query,
                expected_sources=expected_sources,
            )
        )

    return tuple(cases)

def get_source_filename(document: Document) -> str:
    """兼容Windows和Linux路径，只返回用于评估的文件名。"""
    # metadata中的source可能是完整路径，也可能没有提供来源。
    source = str(document.metadata.get("source") or "未知来源")

    # 先统一路径分隔符，再取最后一段文件名。
    return source.replace("\\", "/").rsplit("/", maxsplit=1)[-1]

def search_case(
    vector_store_service: VectorStoreService,
    case: RetrievalCase,
    k: int = TOP_K,
) -> tuple[RetrievalResult, ...]:
    """执行一条真实向量检索，并整理成便于打印和计算的结果。"""
    matches = vector_store_service.search_with_relevance_scores(
        case.query,
        k=k,
    )

    results: list[RetrievalResult] = []

    for rank, (document, score) in enumerate(matches, start=1):
        # 压缩换行和连续空格，避免知识片段在控制台占用太多行。
        preview = " ".join(document.page_content.split())[:80]

        results.append(
            RetrievalResult(
                rank=rank,
                source=get_source_filename(document),
                score=float(score),
                preview=preview,
            )
        )

    # 使用不可变元组返回结果，防止后续计算过程中意外修改排名。
    return tuple(results)

def find_expected_rank(
    case: RetrievalCase,
    results: tuple[RetrievalResult, ...],
) -> int | None:
    """返回预期来源首次出现的排名，没有命中时返回None。"""
    for result in results:
        # 一个正例可以配置多个可接受的知识来源。
        if result.source in case.expected_sources:
            return result.rank

    # 返回None表示正确来源没有进入本次Top-K结果。
    return None

def calculate_summary(
    cases: tuple[RetrievalCase, ...],
    results_by_case: dict[str, tuple[RetrievalResult, ...]],
    k: int = TOP_K,
) -> EvaluationSummary:
    """根据全部检索结果计算Hit@K、MRR和分数边界。"""
    positive_cases = [
        case
        for case in cases
        if case.kind == "positive"
    ]
    negative_cases = [
        case
        for case in cases
        if case.kind == "negative"
    ]

    # 正负例缺少任何一类时，都无法完整评估检索和拒答能力。
    if not positive_cases:
        raise ValueError("至少需要一条正例才能计算检索指标")

    if not negative_cases:
        raise ValueError("至少需要一条负例才能计算分数边界")

    positive_ranks: list[int | None] = []
    positive_top_1_scores: list[float] = []
    negative_top_1_scores: list[float] = []

    for case in positive_cases:
        results = results_by_case.get(case.case_id, ())

        # 没有检索结果通常表示评估流程异常，不能静默跳过。
        if not results:
            raise ValueError(f"正例{case.case_id}没有检索结果")

        positive_ranks.append(
            find_expected_rank(case, results)
        )
        positive_top_1_scores.append(results[0].score)

    for case in negative_cases:
        results = results_by_case.get(case.case_id, ())

        if not results:
            raise ValueError(f"负例{case.case_id}没有检索结果")

        negative_top_1_scores.append(results[0].score)

    # Hit@1表示正确来源排在第一名的正例比例。
    hit_at_1 = sum(
        rank == 1
        for rank in positive_ranks
    ) / len(positive_cases)

    # Hit@K表示正确来源进入前K名的正例比例。
    hit_at_k = sum(
        rank is not None and rank <= k
        for rank in positive_ranks
    ) / len(positive_cases)

    # MRR会对更靠前的正确结果给予更高分，未命中时按0计算。
    mrr = sum(
        1 / rank if rank is not None else 0.0
        for rank in positive_ranks
    ) / len(positive_cases)

    positive_min_score = min(positive_top_1_scores)
    negative_max_score = max(negative_top_1_scores)

    # 正数表示所有已知正例分数都高于负例，负数表示两类分数重叠。
    score_gap = positive_min_score - negative_max_score
    has_score_overlap = score_gap <= 0

    # 分数重叠时不存在能够正确分开两类问题的单一阈值。
    threshold_candidate = (
        None
        if has_score_overlap
        else (positive_min_score + negative_max_score) / 2
    )

    return EvaluationSummary(
        hit_at_1=hit_at_1,
        hit_at_k=hit_at_k,
        mrr=mrr,
        positive_min_top_1_score=positive_min_score,
        negative_max_top_1_score=negative_max_score,
        score_gap=score_gap,
        has_score_overlap=has_score_overlap,
        threshold_candidate=threshold_candidate,
    )

if __name__ == "__main__":
    loaded_cases = load_cases()

    positive_count = sum(
        case.kind == "positive"
        for case in loaded_cases
    )
    negative_count = len(loaded_cases) - positive_count

    print("评估用例加载成功")
    print("用例总数：", len(loaded_cases))
    print("正例数量：", positive_count)
    print("负例数量：", negative_count)

    # 只创建一次服务，后续所有用例复用同一个Chroma连接。
    vector_store_service = VectorStoreService()

    # 使用用例ID保存检索结果，供后续统一计算指标。
    results_by_case: dict[str, tuple[RetrievalResult, ...]] = {}

    for case in loaded_cases:
        print()
        print(f"用例：{case.case_id} | 类型：{case.kind}")
        print(f"问题：{case.query}")
        print(f"预期来源：{', '.join(case.expected_sources) or '无'}")

        # 每条问题只调用一次真实向量检索，打印和计算复用同一份结果。
        case_results = search_case(vector_store_service, case)
        results_by_case[case.case_id] = case_results

        for result in case_results:
            print(
                f"Top-{result.rank} | "
                f"分数：{result.score:.4f} | "
                f"来源：{result.source}"
            )
            print(f"预览：{result.preview}")

    # 所有问题检索完成后，统一计算并展示评估指标。
    summary = calculate_summary(
        loaded_cases,
        results_by_case,
    )

    print()
    print("检索评估汇总")
    print(f"Hit@1：{summary.hit_at_1:.2%}")
    print(f"Hit@{TOP_K}：{summary.hit_at_k:.2%}")
    print(f"MRR：{summary.mrr:.4f}")
    print(
        "正例最低Top-1分数："
        f"{summary.positive_min_top_1_score:.4f}"
    )
    print(
        "负例最高Top-1分数："
        f"{summary.negative_max_top_1_score:.4f}"
    )
    print(
        "正负例分数间隔："
        f"{summary.score_gap:.4f}"
    )
    print(
        "分数是否重叠："
        f"{'是' if summary.has_score_overlap else '否'}"
    )

    if summary.threshold_candidate is None:
        # 分数重叠时明确报告失败，不能输出一个误导性的阈值。
        print("候选拒答阈值：无法仅根据相关性分数确定")
    else:
        print(
            "候选拒答阈值："
            f"{summary.threshold_candidate:.4f}"
        )
