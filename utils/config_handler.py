"""集中读取项目 YAML 配置；敏感信息由环境变量单独管理。"""

import yaml

from utils.path_tool import get_abs_path


def load_rag_config(
    config_path: str = get_abs_path("config/rag.yml"),
    encoding: str = "utf-8",
):
    """读取模型与 Embedding 配置。"""
    with open(config_path, "r", encoding=encoding) as f:
        # safe_load 仅构造基础类型，避免 YAML 实例化任意 Python 对象。
        return yaml.safe_load(f)


def load_chroma_config(
    config_path: str = get_abs_path("config/chroma.yml"),
    encoding: str = "utf-8",
):
    """读取向量库、分片与知识文件配置。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_prompts_config(
    config_path: str = get_abs_path("config/prompts.yml"),
    encoding: str = "utf-8",
):
    """读取 Prompt 文件路径配置。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_agent_config(
    config_path: str = get_abs_path("config/agent.yml"),
    encoding: str = "utf-8",
):
    """读取 Agent 数据源和 Demo 上下文配置。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


# 配置在模块导入时加载一次，供各服务复用。
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
