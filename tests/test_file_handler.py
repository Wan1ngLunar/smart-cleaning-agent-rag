import hashlib
from pathlib import Path

from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
)


def test_get_file_md5_hex(tmp_path):
    test_file = tmp_path / "sample.txt" # 在临时目录创建 sample.txt
    test_file.write_bytes(b"robot-cleaner") # 写入二进制内容：robot-cleaner

    expected = hashlib.md5(b"robot-cleaner").hexdigest() # 原生Python直接计算这段文字的md5，作为标准答案

    assert get_file_md5_hex(str(test_file)) == expected


def test_listdir_with_allowed_type_filters_files(tmp_path):
    # 创建3个文件 guide.txt、manual.pdf、records.csv
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
    missing_directory = tmp_path / "missing" # 构造一个不存在的文件夹 tmp_path/missing,为了边界测试

    result = listdir_with_allowed_type(
        str(missing_directory),
        ("txt", "pdf"),
    )

    assert result == ()