import csv
import json
import os
from datetime import date

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()


external_data: dict[str, dict[str, dict[str, str]]] = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(
    description=(
        "获取指定城市的演示天气数据；该数据不是实时天气，"
        "返回内容会明确标注数据来源"
    )
)
def get_weather(city: str) -> str:
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
    if external_data:
        return

    external_data_path = get_abs_path(agent_conf["external_data_path"])

    if not os.path.isfile(external_data_path):
        raise FileNotFoundError(
            f"外部数据文件 {external_data_path} 不存在"
        )

    loaded_data: dict[str, dict[str, dict[str, str]]] = {}

    with open(
        external_data_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

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

    external_data.update(loaded_data)


@tool(
    description=(
        "从本地演示数据中获取指定用户在指定月份的使用记录，"
        "返回JSON字符串；未检索到时返回空字符串"
    )
)
def fetch_external_data(user_id: str, month: str) -> str:
    load_external_data()

    normalized_user_id = user_id.strip()
    normalized_month = month.strip()

    record = external_data.get(
        normalized_user_id,
        {},
    ).get(normalized_month)

    if record is None:
        logger.warning(
            "[fetch_external_data]未检索到演示使用记录："
            f"用户 {normalized_user_id}，月份 {normalized_month}"
        )
        return ""
    return json.dumps(record, ensure_ascii=False)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
