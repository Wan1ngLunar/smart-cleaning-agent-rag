import os
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import (
    BaseChatModel,
    ChatTongyi,
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path

# 系统环境变量优先；本地 .env 只为尚未设置的变量提供开发默认值。
load_dotenv(
    dotenv_path=get_abs_path(".env"),
    override=False,
)


def get_required_env(name: str) -> str:
    """读取必需环境变量，并在模型初始化前给出可操作的错误提示。"""
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"缺少必要环境变量 {name}。"
            "请复制 .env.example 为 .env，并填写真实值。"
        )

    return value


# 模块加载时执行一次 fail-fast 校验，避免请求发出后才发现密钥缺失。
DASHSCOPE_API_KEY = get_required_env("DASHSCOPE_API_KEY")


class BaseModelFactory(ABC):
    """聊天模型与 Embedding 模型工厂的统一接口。"""

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            api_key=DASHSCOPE_API_KEY,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=DASHSCOPE_API_KEY,
        )


# 在进程内复用模型客户端，避免每次工具调用重复创建连接对象。
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
