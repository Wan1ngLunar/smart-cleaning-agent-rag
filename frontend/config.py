import os

from dotenv import load_dotenv

from utils.path_tool import get_abs_path

# 没有显式配置时，本机开发默认访问8000端口。
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

# 从项目根目录读取.env；系统环境变量仍然拥有更高优先级。
load_dotenv(
    dotenv_path=get_abs_path(".env"),
)


def get_api_base_url() -> str:
    """读取并校验Streamlit使用的FastAPI后端地址。"""
    raw_base_url = os.getenv(
        "API_BASE_URL",
        DEFAULT_API_BASE_URL,
    )
    base_url = raw_base_url.strip().rstrip(
        "/"
    )

    if not base_url:
        raise ValueError(
            "API_BASE_URL不能为空"
        )

    if not base_url.startswith(
        (
            "http://",
            "https://",
        )
    ):
        raise ValueError(
            "API_BASE_URL必须使用http或https协议"
        )

    return base_url
