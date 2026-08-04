import hashlib
from pathlib import Path

from pypdf import PdfWriter

from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
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

def test_txt_loader_reads_utf8_and_keeps_source(tmp_path):
    """TXT加载器应读取UTF-8中文，并保留原始来源路径。"""
    test_file = tmp_path / "中文知识.txt"
    expected_content = "扫地机器人需要定期清理滤网和滚刷。"

    # 直接写入可读中文，验证加载器没有使用错误编码。
    test_file.write_text(
        expected_content,
        encoding="utf-8",
    )

    documents = txt_loader(str(test_file))

    # 一个TXT文件应先转换为一个Document，再交给后续切分器分段。
    assert len(documents) == 1
    assert documents[0].page_content == expected_content
    assert documents[0].metadata == {
        "source": str(test_file),
    }

def test_pdf_loader_splits_pages_and_keeps_metadata(tmp_path):
    """PDF加载器应按页生成文档，并保留来源和页码。"""
    test_file = tmp_path / "测试手册.pdf"
    writer = PdfWriter()

    # 创建两个空白页面，避免测试依赖项目data目录中的真实资料。
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)

    # 以二进制模式写入合法PDF文件。
    with test_file.open("wb") as file:
        writer.write(file)

    documents = pdf_loader(str(test_file))

    # 每个PDF页面应分别转换为一个Document。
    assert len(documents) == 2

    # 空白页提取不到正文时，加载器应返回空字符串而不是None。
    assert documents[0].page_content == ""
    assert documents[1].page_content == ""

    # page是从0开始的内部索引，page_label是从1开始的展示页码。
    assert documents[0].metadata == {
        "source": str(test_file),
        "page": 0,
        "page_label": "1",
    }
    assert documents[1].metadata == {
        "source": str(test_file),
        "page": 1,
        "page_label": "2",
    }
