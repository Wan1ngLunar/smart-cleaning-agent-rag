import argparse

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
        and "参考来源：" in answer
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

    # 整个评估过程复用一个RAG服务和Chroma连接。
    rag_service = RagSummarizeService()
    passed_count = 0

    for case in selected_cases:
        answer = rag_service.rag_summarize(case.query)
        passed = is_case_passed(case, answer)

        if passed:
            passed_count += 1

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
        print("实际输出：")
        print(answer)

    print()
    print(
        "端到端可回答性评估："
        f"{passed_count}/{len(selected_cases)}通过"
    )

    # 任意用例失败时返回非零退出码，便于以后接入自动化流程。
    if passed_count != len(selected_cases):
        raise SystemExit(1)
