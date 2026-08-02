import time

import streamlit as st

from agent.react_agent import ReactAgent
from utils.session_handler import (
    create_thread_id,
    normalize_thread_id,
)


def release_agent(agent: ReactAgent) -> None:
    """Streamlit释放会话资源时关闭Agent的SQLite连接。"""
    agent.close()


@st.cache_resource(
    scope="session",
    show_spinner=False,
    on_release=release_agent,
)
def get_agent() -> ReactAgent:
    """每个浏览器会话复用一个独立的Agent和SQLite连接。"""
    return ReactAgent()


st.title("智扫通机器人智能客服")
st.divider()

agent = get_agent()

# URL中的thread_id允许浏览器刷新或应用重启后定位同一个持久化会话。
raw_thread_id = st.query_params.get(
    "thread_id"
)
thread_id = normalize_thread_id(
    raw_thread_id
)

# 参数缺失、非法或格式不标准时，将安全的新ID写回浏览器地址栏。
if raw_thread_id != thread_id:
    st.query_params["thread_id"] = thread_id

# 首次打开页面或URL切换到另一个会话时，从SQLite恢复页面历史。
if st.session_state.get("thread_id") != thread_id:
    st.session_state["thread_id"] = thread_id
    st.session_state["message"] = agent.get_history(
        thread_id
    )
elif "message" not in st.session_state:
    # 防止局部Session State丢失时页面没有初始化消息列表。
    st.session_state["message"] = agent.get_history(
        thread_id
    )

# 只显示会话ID前8位，完整ID已经保存在URL中。
st.sidebar.caption(
    f"当前会话：{thread_id[:8]}…"
)

if st.sidebar.button("新建对话"):
    new_thread_id = create_thread_id()

    # 同时更新URL、模型会话键和页面消息，避免新旧状态错位。
    st.query_params["thread_id"] = new_thread_id
    st.session_state["thread_id"] = new_thread_id
    st.session_state["message"] = []
    st.rerun()

for message in st.session_state["message"]:
    # 页面历史只包含经过ReactAgent过滤的用户和最终助手文本。
    st.chat_message(
        message["role"]
    ).write(
        message["content"]
    )

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response_messages: list[str] = []

    with st.spinner("智能客服思考中..."):
        response_stream = agent.execute_stream(
            prompt,
            thread_id=thread_id,
        )

        def capture(
            generator,
            cache_list: list[str],
        ):
            """缓存Agent输出，并拆成字符流交给Streamlit渲染。"""
            for chunk in generator:
                cache_list.append(chunk)

                # 逐字符输出只负责页面打字效果，不改变模型或持久化状态。
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message(
            "assistant"
        ).write_stream(
            capture(
                response_stream,
                response_messages,
            )
        )

        if response_messages:
            # Agent可能产生多个中间输出，页面只保存最后的完整回答。
            st.session_state["message"].append(
                {
                    "role": "assistant",
                    "content": response_messages[-1].strip(),
                }
            )
        else:
            # 防止异常空流导致访问response_messages[-1]时报错。
            st.warning(
                "Agent没有返回可展示的文本，请稍后重试。"
            )

        st.rerun()
