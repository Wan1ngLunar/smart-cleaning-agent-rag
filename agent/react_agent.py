from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

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
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    """封装带工具、中间件和会话记忆的 LangGraph Agent。"""

    def __init__(self):
        # InMemorySaver 按 thread_id 保存消息；进程退出后历史会清空，适合本地演示。
        self.checkpointer = InMemorySaver()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.checkpointer,
        )

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

    for chunk in agent.execute_stream(
        "给我生成我的使用报告",
        thread_id="local-debug",
    ):
        print(chunk, end="", flush=True)
