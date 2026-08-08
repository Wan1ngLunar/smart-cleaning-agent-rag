import pytest

from frontend.config import (
    DEFAULT_API_BASE_URL,
    get_api_base_url,
)


def test_api_base_url_uses_local_default(
    monkeypatch,
):
    """环境变量缺失时应使用本机开发地址。"""
    monkeypatch.delenv(
        "API_BASE_URL",
        raising=False,
    )

    assert (
        get_api_base_url()
        == DEFAULT_API_BASE_URL
    )


def test_api_base_url_is_normalized(
    monkeypatch,
):
    """地址两侧空格和末尾斜杠应被清理。"""
    monkeypatch.setenv(
        "API_BASE_URL",
        "  http://api:8000/  ",
    )

    assert (
        get_api_base_url()
        == "http://api:8000"
    )


@pytest.mark.parametrize(
    "invalid_url",
    [
        "",
        "ftp://api:8000",
    ],
)
def test_api_base_url_rejects_invalid_value(
    monkeypatch,
    invalid_url,
):
    """空地址和非HTTP协议不能进入HTTP客户端。"""
    monkeypatch.setenv(
        "API_BASE_URL",
        invalid_url,
    )

    with pytest.raises(ValueError):
        get_api_base_url()
