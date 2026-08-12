from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model.factory import chat_model
from rag.retrieval_pipeline import HybridRerankRetriever
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts

NO_CONTEXT_RESPONSE = (
    "未在本地知识库中检索到足够的参考资料，"
    "暂时无法基于知识库回答该问题。"
)
EMPTY_ANSWER_RESPONSE = "已检索到参考资料，但模型未生成有效回答。"
# 模型判断资料不足时返回该内部标记，最终不会直接展示给用户。
INSUFFICIENT_CONTEXT_MARKER = "__INSUFFICIENT_CONTEXT__"


class RagSummarizeService:
    """检索本地知识库，生成回答并附加可追溯的参考来源。"""

    def __init__(
            self,
            retriever: HybridRerankRetriever | None = None,
    ):
        """初始化检索流水线和回答生成链。"""
        # 正式环境创建真实混合检索流水线；
        # 测试可以注入假检索器，避免调用Chroma和外部模型接口。
        self.retriever = retriever or HybridRerankRetriever()
        self._closed = False

        self.prompt_text = load_rag_prompts()

        # 将知识库边界放入System消息，提高资料充分性约束的优先级。
        self.prompt_template = self._build_prompt_template()
        self.model = chat_model
        self.chain = self._init_chain()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """将知识库规则与用户输入放入不同角色的消息。"""
        return ChatPromptTemplate.from_messages(
            [
                # System消息只保存长期规则，不混入用户问题和检索内容。
                ("system", self.prompt_text),
                (
                    "human",
                    "用户提问：\n{input}\n\n"
                    "参考资料：\n{context}",
                ),
            ]
        )

    def _init_chain(self):
        # 不再打印完整 Prompt，避免用户问题和知识文档内容进入终端日志。
        return self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(
            self,
            query: str,
    ) -> list[Document]:
        """返回经过混合召回和重排序的最终知识片段。"""
        return self.retriever.retrieve(query)

    def close(self) -> None:
        """关闭检索流水线持有的HTTP连接。"""
        if self._closed:
            return

        self.retriever.close()
        self._closed = True

    @staticmethod
    def _format_source(document: Document) -> str:
        """将Document元数据转换为跨平台、面向用户的来源名称。"""
        source = str(document.metadata.get("source") or "未知来源")

        # Chroma可能保存Windows或Linux路径，统一分隔符后只展示文件名。
        filename = source.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        page_label = document.metadata.get("page_label")

        if page_label in (None, ""):
            page = document.metadata.get("page")
            # PyPDFLoader的page从0开始，而面向用户的页码应从1开始.
            if isinstance(page, int):
                page_label = page + 1

        if page_label not in (None, ""):  #  有自带的page_label
            return f"{filename}（第{page_label}页）"

        return filename

    @classmethod
    def _build_context_and_sources(
        cls,
        documents: list[Document],
    ) -> tuple[str, tuple[str, ...]]:
        """构建带来源编号的模型上下文，并按首次出现顺序去重来源。"""
        source_ids: dict[str, int] = {} # key：来源名称（文件名+页码） value：来源唯一编号
        context_parts: list[str] = []  # 存放每一段参考资料文本，最后统一拼接

        for reference_number, document in enumerate(documents, start=1):
            source = cls._format_source(document)

            if source not in source_ids:
                source_ids[source] = len(source_ids) + 1

            source_id = source_ids[source]
            context_parts.append(
                f"【参考资料{reference_number}｜来源[{source_id}]】\n"
                f"{document.page_content.strip()}"
            )

        return "\n\n".join(context_parts), tuple(source_ids)

    @staticmethod
    def _append_sources(answer: str, sources: tuple[str, ...]) -> str:
        """由服务层稳定追加来源列表，不依赖模型自行生成文件名。"""
        source_lines = "\n".join(
            f"[{source_id}] {source}"
            for source_id, source in enumerate(sources, start=1)
        )
        return f"{answer}\n\n参考来源：\n{source_lines}"

    def rag_summarize(self, query: str) -> str:
        """基于有效检索片段回答问题；无资料时不调用模型。"""
        retrieved_documents = self.retriever_docs(query)

        # 流水线已经完成相关性筛选，这里只防御性过滤空白片段。
        context_docs = [
            document
            for document in retrieved_documents
            if document.page_content.strip()
        ]

        if not context_docs:
            # 日志只记录结果数量，不记录可能包含隐私信息的用户原始问题。
            logger.warning(
                "[rag_summarize]检索流水线未返回有效知识片段"
            )
            return NO_CONTEXT_RESPONSE

        context, sources = self._build_context_and_sources(context_docs)
        logger.info(
            f"[rag_summarize]检索到{len(context_docs)}个有效片段，"
            f"来自{len(sources)}个来源"
        )

        answer = str(
            self.chain.invoke(
                {
                    "input": query,
                    "context": context,
                }
            )
        ).strip()

        if answer == INSUFFICIENT_CONTEXT_MARKER:
            # 模型确认资料不能直接回答问题时，不向用户展示内部标记或无关来源。
            logger.warning(
                "[rag_summarize]检索片段与问题相关，但资料不足以直接回答"
            )
            return NO_CONTEXT_RESPONSE

        if not answer:
            logger.warning("[rag_summarize]模型未生成有效回答")
            answer = EMPTY_ANSWER_RESPONSE

        return self._append_sources(answer, sources)


if __name__ == "__main__":
    rag = RagSummarizeService()

    try:
        print(
            rag.rag_summarize(
                "小户型适合哪些扫地机器人"
            )
        )
    finally:
        # 即使模型调用失败，也要释放重排序器的HTTP连接。
        rag.close()
