# Iterator用于明确标注流式方法会逐步产生字符串。
from collections.abc import Iterator
from time import perf_counter
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from agent.tools.agent_tools import (
    close_rag_service,
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import (
    monitor_model,
    monitor_tool,
    report_prompt_switch,
)
from model.factory import chat_model
from utils.checkpoint_handler import create_sqlite_checkpointer
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class AgentExecutionError(RuntimeError):
    """表示Agent请求失败，但只向页面暴露安全且可追踪的信息。"""

    def __init__(self, request_id: str):
        # 保存问题编号，测试和页面都可以读取。
        self.request_id = request_id

        # 页面只展示友好说明和问题编号，不暴露底层异常与调用栈。
        self.public_message = (
            "请求处理暂时失败，请稍后重试。"
            f"问题编号：{request_id}"
        )

        super().__init__(self.public_message)

class ReactAgent:
    """封装带工具、中间件和会话记忆的 LangGraph Agent。"""

    def __init__(self):
        # SQLite按thread_id持久化模型状态，应用重启后仍可恢复同一会话。
        checkpointer, checkpoint_connection = create_sqlite_checkpointer(
            agent_conf["checkpoint_path"]
        )
        self.checkpointer = checkpointer

        # 保存底层连接，应用结束时需要显式关闭文件句柄。
        self._checkpoint_connection = checkpoint_connection
        try:
            self.agent = create_agent(
                model=chat_model,
                system_prompt=load_system_prompts(),
                tools=[
                    rag_summarize,
                    get_weather,
                    get_user_location,
                    get_user_id,
                    get_current_month,
                    fetch_external_data,
                    fill_context_for_report,
                ],
                middleware=[
                    monitor_tool,
                    monitor_model,
                    report_prompt_switch,
                ],
                checkpointer=self.checkpointer,
            )
        except Exception:
            # Agent构建失败时立即释放SQLite连接，避免遗留文件锁。
            self.close()
            raise

    def close(self) -> None:
        """关闭SQLite连接和共享RAG服务；重复调用也不会报错。"""
        if self._checkpoint_connection is not None:
            # 释放LangGraph会话检查点数据库的文件句柄。
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

        # 释放RAG重排序器持有的HTTP连接池。
        # 如果本进程从未调用RAG工具，该函数不会创建或关闭任何服务。
        close_rag_service()

    def get_history(
            self,
            thread_id: str,
    ) -> list[dict[str, str]]:
        """读取指定会话中适合在聊天页面展示的历史消息。"""
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        state = self.agent.get_state(config)
        messages = state.values.get(
            "messages",
            [],
        )

        history: list[dict[str, str]] = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif (
                    isinstance(message, AIMessage)
                    # 带工具调用的AI消息属于中间步骤，不应展示成最终回答。
                    and not message.tool_calls
            ):
                role = "assistant"
            else:
                # ToolMessage和其他内部状态不属于页面聊天记录。
                continue

            # 当前页面只展示纯文本消息，跳过图片等结构化内容。
            if not isinstance(message.content, str):
                continue

            content = message.content.strip()

            if content:
                history.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        return history

    def execute_stream(
        self,
        query: str,
        thread_id: str,
    ) -> Iterator[str]:
        """执行一次流式提问，并记录可追踪的请求生命周期。"""
        # 每次请求生成独立编号，避免使用问题文本作为日志标识。
        request_id = uuid4().hex[:12]
        session_id = thread_id[:8]
        started_at = perf_counter()

        logger.info(
            "[agent_request]请求开始 "
            "request_id=%s session_id=%s",
            request_id,
            session_id,
        )

        input_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                },
            ]
        }

        # 相同thread_id读取历史状态，更换ID则创建隔离会话。
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # 中间件可以读取请求编号，但日志不记录完整会话ID。
        runtime_context = {
            "report": False,
            "request_id": request_id,
            "session_id": session_id,
        }

        try:
            for chunk in self.agent.stream(
                input_dict,
                config=config,
                stream_mode="values",
                context=runtime_context,
            ):
                latest_message = chunk["messages"][-1]

                if latest_message.content:
                    yield (
                        latest_message.content.strip()
                        + "\n"
                    )
        except Exception as error:
            elapsed_ms = (
                perf_counter() - started_at
            ) * 1000

            # exception会把真实异常堆栈写入服务端日志，方便按编号排查。
            logger.exception(
                "[agent_request]请求失败 "
                "request_id=%s session_id=%s "
                "elapsed_ms=%.2f error_type=%s",
                request_id,
                session_id,
                elapsed_ms,
                type(error).__name__,
            )

            # 保留异常因果链供日志排查，页面下一步只展示public_message。
            raise AgentExecutionError(
                request_id
            ) from error

        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000

        logger.info(
            "[agent_request]请求成功 "
            "request_id=%s session_id=%s "
            "elapsed_ms=%.2f",
            request_id,
            session_id,
            elapsed_ms,
        )


if __name__ == "__main__":
    agent = ReactAgent()

    try:
        for chunk in agent.execute_stream(
            "给我生成我的使用报告",
            thread_id="local-debug",
        ):
            print(chunk, end="", flush=True)
    finally:
        # 即使模型调用失败，也要释放SQLite数据库连接。
        agent.close()
