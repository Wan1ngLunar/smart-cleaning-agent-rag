import argparse
import math
import re
from time import perf_counter

from evaluation.evaluate_retrieval import RetrievalCase, load_cases
from rag.rag_service import (
    EMPTY_ANSWER_RESPONSE,
    NO_CONTEXT_RESPONSE,
    RagSummarizeService,
)

# 默认只运行三条代表性用例，降低日常验证的时间和API调用量。
SMOKE_CASE_IDS = (
    "dtof_navigation",
    "unrelated_cooking",
    "handheld_vacuum_repair",
)

# 匹配文末参考来源列表的行：[1] 文件名.pdf
SOURCE_LINE_PATTERN = re.compile(
    r"^\[(\d+)\]\s+(.+?)\s*$",
    re.MULTILINE,
)

# 匹配正文里面行内引用标记：文本[1]文本
INLINE_CITATION_PATTERN = re.compile(
    r"\[(\d+)\]"
)

# 评测标注使用文件名，不包含PDF页码，因此比较前移除展示页码。
PDF_PAGE_SUFFIX_PATTERN = re.compile(
    r"（第[^）]+页）$"
)

def parse_args() -> argparse.Namespace:
    """解析端到端评估的运行模式。"""
    parser = argparse.ArgumentParser(
        description="运行RAG真实模型可回答性评估"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="运行YAML中的全部评估用例",
    )
    return parser.parse_args()

def select_cases(
    loaded_cases: tuple[RetrievalCase, ...],
    run_all: bool,
) -> tuple[RetrievalCase, ...]:
    """根据命令行选项选择冒烟用例或全部用例。"""
    if run_all:
        # 全量模式直接保持YAML中的原始顺序。
        return loaded_cases

    cases_by_id = {
        case.case_id: case
        for case in loaded_cases
    }

    # 冒烟用例缺失时应在调用真实API之前明确报错。
    missing_case_ids = [
        case_id
        for case_id in SMOKE_CASE_IDS
        if case_id not in cases_by_id
    ]

    if missing_case_ids:
        raise ValueError(
            "缺少端到端冒烟用例："
            + ", ".join(missing_case_ids)
        )

    return tuple(
        cases_by_id[case_id]
        for case_id in SMOKE_CASE_IDS
    )

def extract_cited_sources(
    answer: str,
) -> tuple[str, ...]:
    """提取正文实际引用的来源，而非末尾列出的全部来源。"""
    source_heading = "参考来源："

    if source_heading not in answer:
        return ()

    answer_body, source_section = answer.split(
        source_heading,
        maxsplit=1,
    )

    source_by_id: dict[int, str] = {}

    for matched_source in (
        SOURCE_LINE_PATTERN.finditer(
            source_section
        )
    ):
        source_id = int(
            matched_source.group(1)
        )
        displayed_source = (
            matched_source.group(2).strip()
        )

        normalized_source = (
            PDF_PAGE_SUFFIX_PATTERN.sub(
                "",
                displayed_source,
            ).strip()
        )

        if normalized_source:
            source_by_id[source_id] = (
                normalized_source
            )

    # 按正文中首次出现的顺序去重引用编号。
    cited_source_ids = tuple(
        dict.fromkeys(
            int(source_id)
            for source_id in (
                INLINE_CITATION_PATTERN.findall(
                    answer_body
                )
            )
        )
    )

    return tuple(
        source_by_id[source_id]
        for source_id in cited_source_ids
        if source_id in source_by_id
    )


def calculate_nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """使用最近排名法计算百分位数。计算 P95 耗时,95% 请求耗时不超过该值，用来评估接口性能。"""
    if not values:
        raise ValueError(
            "计算百分位数时至少需要一个数值"
        )

    if not 0 < percentile <= 1:
        raise ValueError(
            "percentile必须在0到1之间"
        )

    sorted_values = sorted(values)

    # 最近排名法：P95位置为ceil(样本数乘以0.95)。
    # 百分位数，比如 P95：代表 95% 的数据都小于等于这个数值，只有 5% 的数据比它大。
    # 最近排名法是众多百分位数算法中的其中一种。
    # 公式：
    # 样本从小到大排序
    # rank = ceil(样本数量 × 百分位),ceil是向上取整
    # 取排序后第rank个元素；Python 数组下标从 0，访问rank‑1
    rank = math.ceil(
        len(sorted_values) * percentile
    )

    # Python下标从0开始，因此排名需要减1。
    return sorted_values[rank - 1]

def is_case_passed(
    case: RetrievalCase,
    answer: str,
) -> bool:
    """根据用例类型判断真实RAG结果是否符合预期。"""
    if case.kind == "negative":
        # 负例必须返回统一拒答文本，不能生成知识库答案。
        return answer == NO_CONTEXT_RESPONSE

    # 正例必须生成有效回答，同时附带可追溯来源。
    return (
            answer != NO_CONTEXT_RESPONSE
            and EMPTY_ANSWER_RESPONSE not in answer
            and bool(extract_cited_sources(answer))
    )


if __name__ == "__main__":
    args = parse_args()
    loaded_cases = load_cases()
    selected_cases = select_cases(
        loaded_cases,
        run_all=args.run_all,
    )

    print(
        "评估模式："
        + ("全部用例" if args.run_all else "冒烟用例")
    )
    print("本次用例数量：", len(selected_cases))

    # 整个评估过程复用一个RAG服务、Chroma连接和重排序连接池。
    rag_service = RagSummarizeService()

    passed_count = 0          # 整体用例通过数量
    positive_count = 0        # 正例数量（知识库有答案的问题）
    citation_case_hit_count = 0 # 正例中，至少引用1个正确来源的用例数
    expected_citation_count = 0 # 引用中，引用对的来源个数
    total_citation_count = 0   # Agent总共输出了多少条来源引用
    elapsed_times_ms: list[float] = [] # 保存每一条请求耗时（毫秒），后面算P95

    try:
        for case in selected_cases:
            started_at = perf_counter()

            answer = rag_service.rag_summarize(
                case.query
            )

            elapsed_ms = (
                perf_counter() - started_at
            ) * 1000
            elapsed_times_ms.append(elapsed_ms) # 秒转毫秒，存入耗时列表。

            passed = is_case_passed(
                case,
                answer,
            )

            if passed:
                passed_count += 1

            cited_sources = extract_cited_sources(
                answer
            )

            if case.kind == "positive":
                positive_count += 1
                expected_source_set = set(
                    case.expected_sources
                )

                # 一条正例只要至少引用一个标注来源，就计为用例命中。
                if expected_source_set.intersection(
                    cited_sources
                ):
                    citation_case_hit_count += 1

                # 来源级准确率统计每一条引用是否属于标注来源。
                expected_citation_count += sum(
                    source in expected_source_set
                    for source in cited_sources
                )
                total_citation_count += len(
                    cited_sources
                )

            expected_action = (
                "拒答"
                if case.kind == "negative"
                else "回答并附带来源"
            )
            actual_action = (
                "拒答"
                if answer == NO_CONTEXT_RESPONSE
                else "生成回答"
            )
            status = "通过" if passed else "失败"

            print()
            print(f"用例：{case.case_id}")
            print(f"问题：{case.query}")
            print(f"预期行为：{expected_action}")
            print(f"实际行为：{actual_action}")
            print(f"评估结果：{status}")
            print(f"请求耗时：{elapsed_ms:.2f}毫秒")
            print(
                "引用来源：",
                ", ".join(cited_sources)
                if cited_sources
                else "无",
            )
            print("实际输出：")
            print(answer)
    finally:
        # 即使某条真实模型请求失败，也要释放重排序HTTP连接。
        rag_service.close()

    behavior_accuracy = (
        passed_count / len(selected_cases)
    ) # 行为准确率：整体用例通过率，多少条用例行为符合预期（该拒答拒答，该回答回答）。

    citation_case_hit_rate = (
        citation_case_hit_count / positive_count
        if positive_count
        else 0.0
    ) # 正例预期来源引用命中率：正例中，至少引用 1 个正确来源的用例占比。

    citation_source_accuracy = (
        expected_citation_count
        / total_citation_count
        if total_citation_count
        else 0.0
    ) # 引用来源标注准确率：Agent 输出的所有来源里面，真正属于标准答案来源的比例。

    p95_latency_ms = (
        calculate_nearest_rank_percentile(
            elapsed_times_ms,
            0.95,
        )
    ) # 计算 P95 耗时，95% 的请求耗时不超过该毫秒数。

    print()
    print(
        "端到端可回答性评估："
        f"{passed_count}/{len(selected_cases)}通过"
    )
    print(
        "行为准确率："
        f"{behavior_accuracy:.2%}"
    )
    print(
        "正例预期来源引用命中率："
        f"{citation_case_hit_rate:.2%}"
    )
    print(
        "正文引用标注一致率："
        f"{citation_source_accuracy:.2%}"
    )
    print(
        "P95响应时间："
        f"{p95_latency_ms:.2f}毫秒"
    )

    # 任意用例失败时返回非零退出码，便于接入自动化流程。
    if passed_count != len(selected_cases):
        raise SystemExit(1)
