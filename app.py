import time
import uuid

import streamlit as st

from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# Agent 必须保存在 session_state 中，否则 Streamlit 每次重跑都会丢失内存检查点。
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# thread_id 同时是 LangGraph 会话记忆的隔离键。
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# message 只负责页面展示；模型记忆由 Agent 的 checkpointer 管理。
if "message" not in st.session_state:
    st.session_state["message"] = []

if st.sidebar.button("新建对话"):
    # 同时更换模型会话 ID 和清空页面消息，避免“页面已清空但模型仍记得”。
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message"] = []
    st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])   #  展示历史对话消息

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            thread_id=st.session_state["thread_id"],
        )

        def capture(generator, cache_list):
            """缓存每段 Agent 输出，并拆成字符流交给 Streamlit 渲染。"""
            for chunk in generator:
                cache_list.append(chunk)

                # 逐字符输出仅用于演示打字效果，不影响 Agent 的生成逻辑。
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        # Agent 可能产生多段中间消息，页面历史只保存最后的完整回答。
        st.session_state["message"].append(
            {"role": "assistant", "content": response_messages[-1]}
        )
        st.rerun()  # 刷新后仅展示已保存的对话历史。
