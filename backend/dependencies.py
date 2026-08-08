from typing import Annotated

from fastapi import (
    Depends,
    Request,
)

from agent.react_agent import ReactAgent


#  FastAPI 依赖注入封装，让所有接口路由能简单拿到全局唯一的 ReactAgent，不用每次手动写 request.app.state.agent。
def get_agent(
    request: Request,
) -> ReactAgent:
    """从FastAPI应用状态中取得共享Agent。"""
    return request.app.state.agent


# 端点使用这个类型即可声明“需要共享Agent”。
# FastAPI会自动调用get_agent，不把Agent当成HTTP参数。
AgentDependency = Annotated[
    ReactAgent,
    Depends(get_agent),
] #  Depends是 FastAPI 专属语法：只要函数参数写了这个，FastAPI自动执行 get_agent() 函数，把返回值传给你的接口。
