from pathlib import Path

import yaml

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def load_compose_config() -> dict:
    """读取Compose配置，供架构测试复用。"""
    compose_path = (
        PROJECT_ROOT / "compose.yml"
    )

    return yaml.safe_load(
        compose_path.read_text(
            encoding="utf-8"
        )
    )


def command_text(
    service: dict,
) -> str:
    """把列表或字符串命令统一转换成可检查文本。"""
    command = service["command"]

    if isinstance(command, list):
        return " ".join(
            str(part)
            for part in command
        )

    return str(command)


def test_compose_separates_api_and_web():
    """Compose必须包含职责隔离的后端和前端服务。"""
    config = load_compose_config()
    services = config["services"]

    assert set(services) == {
        "api",
        "web",
    }

    api = services["api"]
    web = services["web"]

    assert "uvicorn" in command_text(api)
    assert "backend.main:app" in (
        command_text(api)
    )
    assert "streamlit" in command_text(
        web
    )
    assert "app.py" in command_text(web)

    assert "8000:8000" in api["ports"]
    assert "8501:8501" in web["ports"]


def test_web_uses_internal_api_without_secret():
    """前端只能获得内部API地址，不能注入模型密钥。"""
    config = load_compose_config()
    services = config["services"]

    api = services["api"]
    web = services["web"]

    assert api["env_file"] == [
        ".env",
    ]
    assert "env_file" not in web
    assert web["environment"] == {
        "API_BASE_URL": "http://api:8000",
    }

    assert web["depends_on"]["api"] == {
        "condition": "service_healthy",
    }


def test_only_api_mounts_runtime_data():
    """数据库、会话和日志目录不能挂载到前端。"""
    config = load_compose_config()
    services = config["services"]

    api_volumes = services["api"][
        "volumes"
    ]

    assert (
        "./storage:/app/storage"
        in api_volumes
    )
    assert "./logs:/app/logs" in (
        api_volumes
    )
    assert "volumes" not in services[
        "web"
    ]


def test_both_services_have_healthchecks():
    """前后端都必须提供独立健康检查。"""
    config = load_compose_config()
    services = config["services"]

    assert "healthcheck" in services[
        "api"
    ]
    assert "healthcheck" in services[
        "web"
    ]


def test_dockerfile_defaults_to_api():
    """公共镜像应暴露双端口并默认启动FastAPI。"""
    dockerfile = (
        PROJECT_ROOT / "Dockerfile"
    ).read_text(
        encoding="utf-8"
    )

    assert "EXPOSE 8000 8501" in (
        dockerfile
    )

    cmd_lines = [
        line
        for line in dockerfile.splitlines()
        if line.startswith("CMD ")
    ]

    assert len(cmd_lines) == 1
    assert "uvicorn" in cmd_lines[0]
    assert "backend.main:app" in (
        cmd_lines[0]
    )
