import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    """负责知识文档切分、入库和检索器创建。"""

    def __init__(self):
        # 配置中的路径先转为项目绝对路径，避免因启动目录不同生成多份数据库。
        persist_directory = get_abs_path(chroma_conf["persist_directory"])
        md5_hex_store = get_abs_path(chroma_conf["md5_hex_store"])

        # 首次运行时自动创建 Chroma 和导入状态所需目录。
        os.makedirs(persist_directory, exist_ok=True)
        os.makedirs(os.path.dirname(md5_hex_store), exist_ok=True)

        # MD5 状态与 Chroma 共用 storage 根目录，防止数据库和导入记录错位。
        self.md5_hex_store = md5_hex_store

        # 显式校验距离度量，避免拼写错误后由Chroma返回难以定位的异常。
        distance_metric = str(
            chroma_conf["distance_metric"]
        ).strip().lower()

        if distance_metric not in {
            "cosine",
            "l2",
            "ip",
        }:
            raise ValueError(
                "distance_metric必须是cosine、l2或ip"
            )

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=persist_directory,
            # 创建HNSW索引时显式使用配置中的距离度量。
            # 已存在集合的度量不会自动改变，因此本次需要重建索引。
            collection_configuration={
                "hnsw": {
                    "space": distance_metric,
                },
            },
        )

        # 使用可配置的重叠分片，减少答案跨分片边界时丢失上下文。
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        """返回固定 top-k 的 Chroma 检索器。"""
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def search_with_relevance_scores(
            self,
            query: str,
            k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """返回文档及相关性分数，分数越高表示与问题越相关。"""
        # 没有单独指定k时，复用Chroma配置中的默认检索数量。
        result_count = chroma_conf["k"] if k is None else k

        # 0或负数没有合理的检索语义，直接报错比静默返回空结果更容易排查。
        if result_count <= 0:
            raise ValueError("k必须是大于0的整数")

        return self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=result_count,
        )

    def load_document(self):
        """读取知识文件、按 MD5 跳过已处理文件，并写入 Chroma。"""

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(self.md5_hex_store):
                # 首次导入先创建空状态文件；此时任何 MD5 都未处理。
                open(self.md5_hex_store, "w", encoding="utf-8").close()
                return False

            with open(self.md5_hex_store, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            with open(self.md5_hex_store, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 文件内容不变时 MD5 不变，可避免重复生成和写入 Embedding。
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 先成功写入向量库，再保存 MD5；写入失败时下次仍可重试。
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)


