import hashlib
from pathlib import Path

from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
)


def test_get_file_md5_hex(tmp_path):
    """项目 MD5 实现应与 Python 标准库结果一致。"""
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"robot-cleaner")

    # 标准库直接计算同一字节串，作为被测函数的期望结果。
    expected = hashlib.md5(b"robot-cleaner").hexdigest()

    assert get_file_md5_hex(str(test_file)) == expected


def test_listdir_with_allowed_type_filters_files(tmp_path):
    """目录扫描只返回允许的 TXT 和 PDF 文件。"""
    # 同时创建允许和不允许的后缀，验证过滤逻辑。
    (tmp_path / "guide.txt").write_text(
        "guide",
        encoding="utf-8",
    )
    (tmp_path / "manual.pdf").write_bytes(b"pdf")
    (tmp_path / "records.csv").write_text(
        "data",
        encoding="utf-8",
    )

    paths = listdir_with_allowed_type(
        str(tmp_path),
        ("txt", "pdf"),
    )
    names = {Path(path).name for path in paths}

    assert names == {"guide.txt", "manual.pdf"}


def test_listdir_with_allowed_type_returns_empty_for_missing_directory(
    tmp_path,
):
    """目录不存在时返回空元组，避免把后缀误当成文件路径。"""
    missing_directory = tmp_path / "missing"

    result = listdir_with_allowed_type(
        str(missing_directory),
        ("txt", "pdf"),
    )

    assert result == ()
