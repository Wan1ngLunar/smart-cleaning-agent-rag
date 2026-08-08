from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import create_app


class FakeAgent:
    """模拟Agent资源，只记录关闭次数。"""

    def __init__(self):
        self.close_count = 0

    def close(self) -> None:
        """记录资源释放，不访问真实SQLite。"""
        self.close_count += 1


def create_test_app() -> tuple[
    FastAPI,
    FakeAgent,
]:
    """创建使用假Agent的测试应用。"""
    fake_agent = FakeAgent()

    application = create_app(
        # 工厂返回固定假对象，避免初始化真实模型和数据库。
        agent_factory=lambda: fake_agent,
    )

    return application, fake_agent


def test_health_endpoint_returns_ok():
    """健康接口应返回稳定且轻量的成功响应。"""
    application, _ = create_test_app()

    # 进入上下文时触发FastAPI启动，退出时触发关闭。
    with TestClient(application) as client:
        response = client.get(
            "/health",
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_openapi_contains_service_metadata():
    """OpenAPI文档应包含可识别的服务名称和版本。"""
    application, _ = create_test_app()

    with TestClient(application) as client:
        response = client.get(
            "/openapi.json",
        )

    assert response.status_code == 200

    openapi_schema = response.json()

    assert (
        openapi_schema["info"]["title"]
        == "智扫通机器人智能客服API"
    )
    assert openapi_schema["info"]["version"] == "1.0.0"
    assert "/health" in openapi_schema["paths"]


def test_health_endpoint_rejects_post_method():
    """健康接口只允许读取，不能接受POST请求。"""
    application, _ = create_test_app()

    with TestClient(application) as client:
        response = client.post(
            "/health",
        )

    assert response.status_code == 405


def test_lifespan_creates_and_closes_agent_once():
    """应用应延迟创建Agent，并在关闭时只释放一次。"""
    created_agents: list[FakeAgent] = []

    def create_fake_agent() -> FakeAgent:
        """记录生命周期实际创建的假Agent。"""
        fake_agent = FakeAgent()
        created_agents.append(fake_agent)
        return fake_agent

    application = create_app(
        agent_factory=create_fake_agent,
    )

    # 仅创建FastAPI对象时，还不应该初始化昂贵的Agent资源。
    assert created_agents == []

    with TestClient(application):
        assert len(created_agents) == 1
        assert (
            application.state.agent
            is created_agents[0]
        )
        assert created_agents[0].close_count == 0

    # 离开TestClient上下文后，应自动关闭同一个Agent。
    assert created_agents[0].close_count == 1
