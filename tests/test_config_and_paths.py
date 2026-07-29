from pathlib import Path

from utils.config_handler import agent_conf, chroma_conf
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
