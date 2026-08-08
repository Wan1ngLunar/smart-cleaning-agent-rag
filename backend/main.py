from collections.abc import (
    AsyncIterator,
    Callable,
)
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.react_agent import ReactAgent
from backend.routers.chat import (
    router as chat_router,
)
from backend.routers.sessions import (
    router as sessions_router,
)
from backend.schemas import HealthResponse

# 工厂函数负责创建Agent。
# 单独定义类型便于测试时传入假Agent，避免访问真实模型和数据库。
AgentFactory = Callable[[], ReactAgent]

# FastAPI 服务入口文件
# 1. 通过 lifespan 生命周期管理器统一管控 ReactAgent 实例：服务启动时创建Agent并挂载至全局state，服务停止时自动关闭释放SQLite会话连接，避免数据库文件锁残留
# 2. 使用工厂模式 AgentFactory 注入Agent构造逻辑，生产环境使用真实ReactAgent，单元测试可传入Mock假Agent，隔离数据库、大模型外部依赖
# 3. 内置 /health 健康探测接口，仅校验Web服务存活，不依赖向量库与LLM，用于容器存活探针
# 4. 提供 create_app 工厂函数解耦应用初始化逻辑，方便自动化测试复用与扩展新增路由
def create_app(
    agent_factory: AgentFactory = ReactAgent,
) -> FastAPI:
    """创建FastAPI应用，并统一管理Agent资源生命周期。"""

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        """在服务启动时创建Agent，关闭时释放SQLite连接。"""
        agent = agent_factory()

        # 把Agent保存在应用状态中，后续所有API请求复用同一实例。
        application.state.agent = agent

        try:
            # yield之前是启动阶段，之后FastAPI开始接受请求。
            yield
        finally:
            # 无论正常关闭还是启动后的运行异常，都释放SQLite连接。
            agent.close()

    application = FastAPI(
        title="智扫通机器人智能客服API",
        description=(
            "为Streamlit前端提供会话历史和Agent问答接口。"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # 注册Agent聊天接口。
    application.include_router(
        chat_router
    )

    # 注册带版本号的会话路由。
    application.include_router(
        sessions_router
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["系统"],
        summary="检查后端服务是否正常",
    )
    def health() -> HealthResponse:
        """返回轻量健康状态，不调用模型或向量数据库。"""
        return HealthResponse()

    return application


# Uvicorn导入该对象后，由lifespan负责创建和关闭真实Agent。
app = create_app()
