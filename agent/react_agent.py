from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import chat_model
from utils.checkpoint_handler import create_sqlite_checkpointer
from utils.config_handler import agent_conf
from utils.prompt_loader import load_system_prompts


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
                    log_before_model,
                    report_prompt_switch,
                ],
                checkpointer=self.checkpointer,
            )
        except Exception:
            # Agent构建失败时立即释放SQLite连接，避免遗留文件锁。
            self.close()
            raise

    def close(self) -> None:
        """关闭SQLite连接；重复调用也不会报错。"""
        if self._checkpoint_connection is None:
            return

        self._checkpoint_connection.close()
        self._checkpoint_connection = None

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

    def execute_stream(self, query: str, thread_id: str):
        """在指定会话中执行一次提问，并逐步产出 Agent 的文本消息。"""
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 相同 thread_id 会读取之前的消息；更换 ID 即创建隔离的新会话。
        config = {"configurable": {"thread_id": thread_id}}

        # report 是单次请求的运行时标记，中间件可将其切换为 True 以启用报告 Prompt。
        for chunk in self.agent.stream(
            input_dict,
            config=config,
            stream_mode="values",
            context={"report": False},
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


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
