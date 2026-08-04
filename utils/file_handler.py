import hashlib
import os
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from utils.logger_handler import logger


def get_file_md5_hex(filepath: str) -> str | None:
    """分块计算文件 MD5；文件无效或读取失败时返回 None。"""
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()

    # 每次只读取 4 KB，避免大文件一次性载入内存。
    chunk_size = 4096
    try:
        # 哈希必须基于原始字节计算，不能使用会改变换行或编码的文本模式。
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(
    path: str,
    allowed_types: tuple[str, ...],
) -> tuple[str, ...]:
    """返回目录内符合后缀要求的文件路径。"""
    files: list[str] = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        # 空元组表示没有文件；不能返回 allowed_types，否则后缀会被当成路径。
        return ()

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(
    filepath: str,
    passwd: str | bytes | None = None,
) -> list[Document]:
    """按页读取PDF，并保留来源路径与页码元数据。"""
    reader = PdfReader(
        filepath,
        password=passwd,
    )
    documents: list[Document] = []

    # page_labels可保留PDF自身的页码标签；普通PDF通常为1、2、3等。
    page_labels = reader.page_labels or []

    for page_index, page in enumerate(reader.pages):
        # LangChain约定page使用从0开始的索引，展示时使用page_label。
        page_label = (
            page_labels[page_index]
            if page_index < len(page_labels)
            else str(page_index + 1)
        )

        documents.append(
            Document(
                # 扫描件或空白页可能提取不到文字，此时使用空字符串。
                page_content=page.extract_text() or "",
                metadata={
                    "source": filepath,
                    "page": page_index,
                    "page_label": page_label,
                },
            )
        )

    return documents


def txt_loader(filepath: str) -> list[Document]:
    """以UTF-8读取TXT，并转换为一个带来源路径的文档。"""
    content = Path(filepath).read_text(encoding="utf-8")

    return [
        Document(
            page_content=content,
            metadata={"source": filepath},
        )
    ]
