import csv
import json
import os
from datetime import date
from threading import Lock

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# RAG服务采用进程内延迟初始化：
# 导入模块时不连接Chroma和重排序接口，第一次调用工具时才创建。
_rag_service: RagSummarizeService | None = None

# FastAPI可能并发处理请求，使用锁避免同时创建多个RAG服务。
_rag_service_lock = Lock()

def get_rag_service() -> RagSummarizeService:
    """延迟创建并复用进程内唯一的RAG服务。"""
    global _rag_service

    # 已经创建时直接返回，避免每次工具调用都重新加锁。
    if _rag_service is not None:
        return _rag_service

    # 首次并发调用时，只允许一个线程执行初始化。
    with _rag_service_lock:
        # 获取锁后必须再次判断，防止等待期间其他线程已经完成创建。
        if _rag_service is None:
            _rag_service = RagSummarizeService()

        return _rag_service


def close_rag_service() -> None:
    """关闭并清空进程内RAG服务；重复调用不会报错。"""
    global _rag_service

    with _rag_service_lock:
        service = _rag_service

        # 把当前_rag_service保存到局部变量service
        # 立刻把全局变量_rag_service = None
        # 释放锁！关闭资源的逻辑放到锁外面执行
        _rag_service = None

    if service is not None:
        # 关闭qwen3-rerank客户端持有的HTTP连接池。
        service.close()

# 缓存结构：用户 ID -> 月份 -> 报告字段 -> 字段内容。
external_data: dict[str, dict[str, dict[str, str]]] = {}


@tool(
    description=(
        "从本地知识库执行向量与BM25混合检索，"
        "经过RRF融合和重排序后生成带来源的回答"
    )
)
def rag_summarize(query: str) -> str:
    """延迟取得共享RAG服务并执行知识库问答。"""
    return get_rag_service().rag_summarize(query)


@tool(
    description=(
        "获取指定城市的演示天气数据；该数据不是实时天气，"
        "返回内容会明确标注数据来源"
    )
)
def get_weather(city: str) -> str:
    # 天气来自明确标注的 Demo 配置，不能在回答中声称为实时查询。
    weather = agent_conf["demo"]["weather"]

    return (
        f"[演示天气数据，非实时查询] "
        f"城市{city}天气为{weather['condition']}，"
        f"气温{weather['temperature_c']}摄氏度，"
        f"空气湿度{weather['humidity_percent']}%，"
        f"{weather['wind']}，"
        f"AQI {weather['aqi']}，"
        f"最近6小时降雨概率{weather['rain_probability']}"
    )


@tool(description="获取当前演示用户所在城市，以纯字符串形式返回")
def get_user_location() -> str:
    return str(agent_conf["demo"]["user_location"])


@tool(description="获取当前演示用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return str(agent_conf["demo"]["user_id"])


@tool(description="获取系统当前月份，以YYYY-MM格式的纯字符串返回")
def get_current_month() -> str:
    return date.today().strftime("%Y-%m")


def load_external_data() -> None:
    """一次性加载并校验本地 CSV 演示数据。"""
    # 已成功加载时直接复用缓存，避免每次报告请求重复读取磁盘。
    if external_data:
        return

    external_data_path = get_abs_path(agent_conf["external_data_path"])

    if not os.path.isfile(external_data_path):
        raise FileNotFoundError(
            f"外部数据文件 {external_data_path} 不存在"
        )

    # 先写入局部字典；只有完整解析成功后才更新全局缓存，避免留下半成品。
    loaded_data: dict[str, dict[str, dict[str, str]]] = {}

    with open(
        external_data_path,
        "r",
        # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 CSV。
        encoding="utf-8-sig",
        # csv 模块要求 newline=""，以正确处理 Windows 换行和带引号的字段。
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        # 按列名校验数据契约，避免列顺序变化或缺列时静默读错。
        required_columns = {
            "用户ID",
            "特征",
            "清洁效率",
            "耗材",
            "对比",
            "时间",
        }
        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"外部数据文件缺少必要列：{missing_text}"
            )

        # 第一行是表头，因此数据行号从 2 开始，报错位置与原 CSV 一致。
        for row_number, row in enumerate(reader, start=2):
            user_id = (row.get("用户ID") or "").strip()
            month = (row.get("时间") or "").strip()

            if not user_id or not month:
                logger.warning(
                    f"[load_external_data]第 {row_number} 行"
                    "缺少用户ID或月份，已跳过"
                )
                continue

            user_records = loaded_data.setdefault(user_id, {})

            # 同一用户同一月份只能有一条记录，否则无法判断哪条是权威数据。
            if month in user_records:
                raise ValueError(
                    f"外部数据第 {row_number} 行存在重复记录："
                    f"用户 {user_id}，月份 {month}"
                )

            user_records[month] = {
                "特征": (row.get("特征") or "").strip(),
                "效率": (row.get("清洁效率") or "").strip(),
                "耗材": (row.get("耗材") or "").strip(),
                "对比": (row.get("对比") or "").strip(),
            }

    if not loaded_data:
        raise ValueError("外部数据文件中没有有效记录")

    # 到这里说明整份文件校验通过，再一次性发布到进程缓存。
    external_data.update(loaded_data)


@tool(
    description=(
        "从本地演示数据中获取指定用户在指定月份的使用记录，"
        "返回JSON字符串；未检索到时返回空字符串"
    )
)
def fetch_external_data(user_id: str, month: str) -> str:
    load_external_data()

    # 工具参数可能由模型生成，先清理首尾空格再查询。
    normalized_user_id = user_id.strip()
    normalized_month = month.strip()

    record = external_data.get(
        normalized_user_id,
        {},
    ).get(normalized_month)

    if record is None:
        if record is None:
            # 只记录查询未命中，不把用户ID和月份等工具参数写入日志。
            logger.warning(
                "[fetch_external_data]未检索到匹配的演示使用记录"
            )
            return ""
        return ""

    # LangChain 工具描述约定返回字符串，因此把内部字典序列化为中文 JSON。
    return json.dumps(record, ensure_ascii=False)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
