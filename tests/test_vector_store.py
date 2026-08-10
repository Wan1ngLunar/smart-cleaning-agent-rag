import pytest

import rag.vector_store as vector_store_module


def test_vector_store_passes_cosine_configuration(
    tmp_path,
    monkeypatch,
):
    """向量服务应把余弦距离配置真正传给Chroma构造器。"""
    captured_kwargs = {}

    class FakeChroma:
        """记录构造参数，避免单元测试创建真实Chroma数据库。"""

        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    # 使用pytest临时目录，避免测试读写真实storage。
    monkeypatch.setitem(
        vector_store_module.chroma_conf,
        "persist_directory",
        str(tmp_path / "chroma"),
    )
    monkeypatch.setitem(
        vector_store_module.chroma_conf,
        "md5_hex_store",
        str(tmp_path / "ingested_md5.txt"),
    )
    monkeypatch.setitem(
        vector_store_module.chroma_conf,
        "distance_metric",
        "cosine",
    )

    # 用假对象替换Chroma，只检查服务传入的参数。
    monkeypatch.setattr(
        vector_store_module,
        "Chroma",
        FakeChroma,
    )

    vector_store_module.VectorStoreService()

    assert captured_kwargs[
        "collection_configuration"
    ] == {
        "hnsw": {
            "space": "cosine",
        },
    }

def test_get_all_documents_rebuilds_langchain_documents():
    """Chroma数据应按相同下标重新组装成LangChain文档。"""

    class StubChroma:
        """返回固定集合数据，不读取真实向量数据库。"""

        def __init__(self):
            self.requested_include: list[str] | None = None

        def get(
            self,
            *,
            include: list[str],
        ) -> dict:
            self.requested_include = include

            return {
                "ids": [
                    "document-1",
                    "document-2",
                ],
                "documents": [
                    "电量低于20%时自动回充。",
                    "滤网清洗后需要完全晾干。",
                ],
                "metadatas": [
                    {"source": "使用说明.txt"},
                    None,
                ],
            }

    # 绕过生产初始化，避免测试创建真实Chroma和模型客户端。
    service = (
        vector_store_module.VectorStoreService.__new__(
            vector_store_module.VectorStoreService
        )
    )
    stub_chroma = StubChroma()
    service.vector_store = stub_chroma

    documents = service.get_all_documents()

    assert stub_chroma.requested_include == [
        "documents",
        "metadatas",
    ]
    assert [document.id for document in documents] == [
        "document-1",
        "document-2",
    ]
    assert documents[0].page_content == (
        "电量低于20%时自动回充。"
    )
    assert documents[0].metadata == {
        "source": "使用说明.txt"
    }
    assert documents[1].metadata == {}


def test_get_all_documents_returns_empty_list_for_empty_collection():
    """空Chroma集合应正常返回空列表。"""

    class EmptyChroma:
        """模拟尚未导入任何知识片段的Chroma集合。"""

        def get(
            self,
            *,
            include: list[str],
        ) -> dict:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }

    service = (
        vector_store_module.VectorStoreService.__new__(
            vector_store_module.VectorStoreService
        )
    )
    service.vector_store = EmptyChroma()

    assert service.get_all_documents() == []


def test_get_all_documents_rejects_misaligned_collection_data():
    """ID、正文和元数据数量不一致时应立即报错。"""

    class InvalidChroma:
        """模拟内部数据数量不一致的异常返回。"""

        def get(
            self,
            *,
            include: list[str],
        ) -> dict:
            return {
                "ids": ["document-1"],
                "documents": [],
                "metadatas": [{}],
            }

    service = (
        vector_store_module.VectorStoreService.__new__(
            vector_store_module.VectorStoreService
        )
    )
    service.vector_store = InvalidChroma()

    with pytest.raises(
        ValueError,
        match="ID、正文和元数据数量不一致",
    ):
        service.get_all_documents()
