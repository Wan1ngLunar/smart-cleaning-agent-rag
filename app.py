import time
from collections.abc import Iterator

import streamlit as st

from frontend.api_client import (
    ApiClientError,
    BackendApiClient,
)
from frontend.config import get_api_base_url
from utils.session_handler import (
    create_thread_id,
    normalize_thread_id,
)

#这是浏览器前端页面代码，不直接连数据库、不运行 Agent、不调用向量库；
#只通过 BackendApiClient HTTP 请求访问 FastAPI 后端服务，
#实现对话、历史记录、新建会话、打字动画效果。

def release_api_client(
    api_client: BackendApiClient,
) -> None:
    """Streamlit释放会话资源时关闭HTTP连接池。"""
    api_client.close()


@st.cache_resource(
    scope="session", # 同一个浏览器标签页共用 1 个客户端；新开标签页才会新建另一个客户端。
    show_spinner=False, # 拿到客户端时不显示加载转圈。
    on_release=release_api_client, # 会话销毁时自动执行释放函数，关闭 http 连接。
)
def get_api_client() -> BackendApiClient:
    """每个浏览器会话复用一个HTTPX连接池。"""
    return BackendApiClient(
        base_url=get_api_base_url(),
    )


def load_history(
    api_client: BackendApiClient,
    thread_id: str,
) -> tuple[
    list[dict[str, str]],
    bool,
]:
    """通过FastAPI读取历史，并返回是否成功。"""
    try:
        messages = api_client.get_history(
            thread_id
        )
    except ApiClientError as error:
        # 页面只展示客户端整理后的安全说明。
        st.error(
            error.public_message
        )
        return [], False

    return messages, True


def typewriter(
    text: str,
) -> Iterator[str]:
    """把完整API回答拆成字符流，仅用于页面打字效果。"""
    for char in text:
        time.sleep(0.01)
        yield char


st.title("智扫通机器人智能客服")
st.divider()

# 前端只创建HTTP客户端，不再创建ReactAgent。
api_client = get_api_client()

# URL中的thread_id用于定位FastAPI后端保存的SQLite会话。
raw_thread_id = st.query_params.get(
    "thread_id"
)
thread_id = normalize_thread_id(
    raw_thread_id
)

# 非法或缺失的ID会被替换成安全的新UUID。
if raw_thread_id != thread_id:
    st.query_params["thread_id"] = thread_id

# 如果缓存里存的会话ID 和 URL里的ID不一致（切换会话/第一次打开页面）
if st.session_state.get(
    "thread_id"
) != thread_id:
    # 调用后端加载历史对话
    history, history_loaded = load_history(
        api_client,
        thread_id,
    )
    st.session_state["message"] = history

    # 加载成功才更新缓存thread_id；失败下次刷新会重新加载
    if history_loaded:
        st.session_state[
            "thread_id"
        ] = thread_id
# 缓存里没有message对话列表，同样加载历史
elif "message" not in st.session_state:
    history, history_loaded = load_history(
        api_client,
        thread_id,
    )
    st.session_state["message"] = history

    if history_loaded:
        st.session_state[
            "thread_id"
        ] = thread_id

# 页面只显示会话ID前8位，完整ID保留在URL中。
st.sidebar.caption(
    f"当前会话：{thread_id[:8]}…"
)

if st.sidebar.button("新建对话"):
    new_thread_id = create_thread_id()

    # 新会话只切换UUID；后端首次收到问题时创建持久化状态。
    st.query_params["thread_id"] = (
        new_thread_id
    )
    st.session_state["thread_id"] = (
        new_thread_id
    )
    st.session_state["message"] = []
    st.rerun()

for message in st.session_state["message"]:
    # 历史已经经过后端和客户端两层公开消息校验。
    st.chat_message(
        message["role"]
    ).write(
        message["content"]
    )

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(
        prompt
    )
    st.session_state["message"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    assistant_message = st.chat_message(
        "assistant"
    )

    with st.spinner(
        "智能客服思考中..."
    ):
        try:
            # Streamlit只发送HTTP请求，不直接执行Agent。
            answer = api_client.chat(
                thread_id=thread_id,
                message=prompt,
            )
        except ApiClientError as error:
            assistant_message.error(
                error.public_message
            )
        else:
            # 后端已经返回完整回答，这里只模拟页面打字效果。
            assistant_message.write_stream(
                typewriter(answer)
            )
            st.session_state[
                "message"
            ].append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # 正常完成后刷新，并通过历史接口验证持久化结果。
            st.rerun()
