import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

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


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
