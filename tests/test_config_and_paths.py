from pathlib import Path

from utils.config_handler import agent_conf, chroma_conf
from utils.path_tool import get_abs_path, get_project_root


def test_project_root_points_to_repository_root():
    expected_root = Path(__file__).resolve().parents[1]

    assert Path(get_project_root()).resolve() == expected_root


def test_runtime_paths_are_based_on_project_root():
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
    assert str(agent_conf["demo"]["user_id"]) == "1001"
    assert agent_conf["demo"]["user_location"] == "\u6df1\u5733"
    assert agent_conf["demo"]["weather"]["condition"] == "\u6674\u5929"