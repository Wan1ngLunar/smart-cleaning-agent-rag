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
