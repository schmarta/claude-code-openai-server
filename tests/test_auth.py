"""Startup safety interlock + bearer-token auth on the /v1 surface.

The server runs the Claude CLI with permission_mode=bypassPermissions, so an
open network bind is RCE. create_app() must refuse a non-loopback bind unless an
api_key is set, and — when one is set — must reject /v1 requests lacking a valid
bearer token. The MCP mount and health routes stay open.
"""

import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.config import Settings


def _settings(**over):
    base = {"host": "127.0.0.1"}
    base.update(over)
    return Settings(**base)


def test_refuses_open_nonloopback(monkeypatch):
    monkeypatch.setattr(main_mod, "get_settings", lambda: _settings(host="0.0.0.0", api_key=None))
    with pytest.raises(RuntimeError):
        main_mod.create_app()


def test_allows_nonloopback_with_key(monkeypatch):
    monkeypatch.setattr(main_mod, "get_settings", lambda: _settings(host="0.0.0.0", api_key="secret"))
    assert main_mod.create_app() is not None


def test_loopback_open_when_no_key(monkeypatch):
    monkeypatch.setattr(main_mod, "get_settings", lambda: _settings(api_key=None))
    client = TestClient(main_mod.create_app())
    assert client.get("/v1/models").status_code == 200


def test_v1_requires_bearer_when_key_set(monkeypatch):
    monkeypatch.setattr(main_mod, "get_settings", lambda: _settings(api_key="secret"))
    client = TestClient(main_mod.create_app())
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_health_open_even_with_key(monkeypatch):
    # systemd / load balancers must still reach health without a token.
    monkeypatch.setattr(main_mod, "get_settings", lambda: _settings(api_key="secret"))
    client = TestClient(main_mod.create_app())
    assert client.get("/healthz").status_code == 200
