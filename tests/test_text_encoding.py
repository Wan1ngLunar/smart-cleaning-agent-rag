import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 只检查应由开发者维护的文本格式，PDF、SQLite 等二进制文件不会被读取。
TEXT_SUFFIXES = {
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".md",
    ".csv",
    ".toml",
}

# 这些目录只包含环境、缓存或运行产物，不属于仓库源码。
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "storage",
    "local_backup",
    "logs",
}

# 分开拼接反斜杠和字母 u，避免本测试源码本身包含被禁止的转义形式。
UNICODE_ESCAPE_PATTERN = re.compile(
    re.escape("\\") + "u" + r"[0-9a-fA-F]{4}"
)


def iter_project_text_files():
    """遍历仓库中需要执行 UTF-8 和可读性检查的文本文件。"""
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(
            directory_name in EXCLUDED_DIRECTORY_NAMES
            for directory_name in path.parts
        ):
            continue

        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def test_project_text_files_are_valid_utf8():
    """所有项目文本文件都必须能够按严格 UTF-8 解码。"""
    for path in iter_project_text_files():
        path.read_text(
            encoding="utf-8",
            errors="strict",
        )


def test_project_text_files_do_not_use_unicode_escape_sequences():
    """源码和数据应直接使用可读中文，而不是 Unicode 转义序列。"""
    for path in iter_project_text_files():
        content = path.read_text(encoding="utf-8")
        match = UNICODE_ESCAPE_PATTERN.search(content)

        assert match is None, (
            f"{path.relative_to(PROJECT_ROOT)} 包含 Unicode 转义："
            f"{match.group() if match else ''}"
        )
