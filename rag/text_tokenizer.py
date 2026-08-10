import re
import unicodedata

"""
这是一段无第三方分词库（jieba/THUDM 等） 的轻量文本切分函数，专门给 BM25 全文检索做预处理：
统一文本格式（全角转半角、小写）
把文本拆成三类片段：中文词语、带单位数字、英文型号
中文用二元组 bigram切分（两字一组），英文 / 数字完整保留不拆分
输出标准化词元列表，供 BM25 做相似度检索
"""
# 匹配连续中文、带单位的数字以及英文技术词。
# 例如：“低电量”“20%”“2cm”“HEPA”“qwen3-max”。
_TOKEN_PATTERN = re.compile(
    r"[一-龥]+"
    r"|\d+(?:\.\d+)?(?:%|[a-z]+)?"
    r"|[a-z]+[a-z0-9._+-]*"
)

# 用于判断当前片段是否全部由常用中文字符组成。
_CHINESE_TEXT_PATTERN = re.compile(r"^[一-龥]+$")


def _create_chinese_bigrams(text: str) -> list[str]:
    """将连续中文切成相邻的两个字符，单字则直接保留。"""
    if len(text) <= 1:
        return [text]

    # “自动回充”会得到“自动”“动回”“回充”。
    # 这种方式不依赖外部分词词典，也能匹配领域术语。
    return [
        text[index : index + 2]
        for index in range(len(text) - 1)
    ]


def tokenize_for_bm25(text: str) -> list[str]:
    """将文本转换成适合BM25检索的确定性词元列表。"""
    # NFKC会把全角字母、数字和百分号转成等价的半角形式；
    # 它不会把正常中文改成Unicode转义字符串。
    normalized_text = unicodedata.normalize(
        "NFKC",
        text,
    ).lower()

    tokens: list[str] = []

    # findall按原文顺序提取中文、数字和英文片段。
    for segment in _TOKEN_PATTERN.findall(normalized_text):
        if _CHINESE_TEXT_PATTERN.fullmatch(segment):
            tokens.extend(
                _create_chinese_bigrams(segment)
            )
        else:
            # 英文、数字和单位整体保留，避免将20%或2cm拆散。
            tokens.append(segment)

    return tokens
