from rag.text_tokenizer import tokenize_for_bm25


def test_tokenize_for_bm25_creates_chinese_bigrams():
    """连续中文应转换成相邻双字词元。"""
    tokens = tokenize_for_bm25("低电量自动回充")

    assert tokens == [
        "低电",
        "电量",
        "量自",
        "自动",
        "动回",
        "回充",
    ]


def test_tokenize_for_bm25_preserves_english_numbers_and_units():
    """英文应转成小写，数字、百分号和单位应整体保留。"""
    tokens = tokenize_for_bm25(
        "HEPA H13，电量低于２０％，门槛2cm"
    )

    assert tokens == [
        "hepa",
        "h13",
        "电量",
        "量低",
        "低于",
        "20%",
        "门槛",
        "2cm",
    ]


def test_tokenize_for_bm25_does_not_cross_punctuation():
    """不同标点片段之间不应生成跨片段的中文双字。"""
    tokens = tokenize_for_bm25("滤网，清洗")

    assert tokens == [
        "滤网",
        "清洗",
    ]
    assert "网清" not in tokens


def test_tokenize_for_bm25_handles_single_character_and_empty_text():
    """中文单字应保留，空白文本应返回空列表。"""
    assert tokenize_for_bm25("水") == ["水"]
    assert tokenize_for_bm25("   ") == []
