from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from langgraph.checkpoint.memory import InMemorySaver


class ReactAgent:
    def __init__(self):
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
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        config = {"configurable": {"thread_id": thread_id}}

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict,
                                       config=config,
                                       stream_mode="values",
                                       context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream(
            "给我生成我的使用报告",
            thread_id="local-debug",
    ):
        print(chunk, end="", flush=True)
