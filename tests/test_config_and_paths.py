from pathlib import Path

from utils.config_handler import (
    agent_conf,
    chroma_conf,
    rag_conf,
)
from utils.path_tool import get_abs_path, get_project_root


def test_project_root_points_to_repository_root():
    """工具计算的项目根目录应与测试文件的仓库根目录一致。"""
    expected_root = Path(__file__).resolve().parents[1]

    assert Path(get_project_root()).resolve() == expected_root


def test_runtime_paths_are_based_on_project_root():
    """运行数据必须落在唯一的项目 storage 目录，不能依赖当前工作目录。"""
    project_root = Path(get_project_root()).resolve()

    chroma_path = Path(
        get_abs_path(chroma_conf["persist_directory"])
    ).resolve()
    md5_path = Path(
        get_abs_path(chroma_conf["md5_hex_store"])
    ).resolve()

    assert chroma_path == project_root / "storage" / "chroma"
    assert md5_path == project_root / "storage" / "ingested_md5.txt"


def test_demo_configuration_is_deterministic():
    """Demo 配置必须保持固定，防止恢复成随机用户和随机天气。"""
    assert str(agent_conf["demo"]["user_id"]) == "1001"
    assert agent_conf["demo"]["user_location"] == "深圳"
    assert agent_conf["demo"]["weather"]["condition"] == "晴天"

def test_chroma_distance_metric_is_cosine():
    """文本向量库应显式使用余弦距离，不能退回默认L2。"""
    assert chroma_conf["distance_metric"] == "cosine"

def test_hybrid_retrieval_config_uses_evaluation_winner():
    """混合检索应使用30组参数实验选出的最优配置。"""
    assert chroma_conf["hybrid_retrieval"] == {
        "vector_candidate_k": 10,
        "bm25_candidate_k": 20,
        "rrf_constant": 10,
    }

def test_rerank_config_covers_hybrid_candidates():
    """重排序配置应覆盖困难候选，并最终恢复现有Top-3。"""
    rerank_config = rag_conf["rerank"]

    assert rerank_config["model_name"] == (
        "qwen3-rerank"
    )
    assert rerank_config["candidate_k"] == 10
    assert rerank_config["top_n"] == (
        chroma_conf["k"]
    )
    assert (
        rerank_config["candidate_k"]
        > rerank_config["top_n"]
    )
    assert (
        rerank_config["timeout_seconds"]
        > 0
    )
    assert rerank_config["instruct"]
