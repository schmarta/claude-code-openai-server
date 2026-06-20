"""Tests for the Ollama / llama.cpp compatibility probe endpoints.

Uses a bare TestClient (no ``with`` block) so the app lifespan — which starts the
MCP bridge and conversation manager — never runs; these handlers only need
``app.state.settings``, which ``create_app`` sets synchronously.
"""

from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def test_api_tags():
    r = client.get("/api/tags")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert body["models"], "expected at least one advertised model"
    first = body["models"][0]
    for key in ("name", "model", "modified_at", "size", "digest", "details"):
        assert key in first


def test_api_show_known_model():
    r = client.post("/api/show", json={"model": "opus"})
    assert r.status_code == 200
    body = r.json()
    for key in ("details", "model_info", "capabilities"):
        assert key in body
    assert "completion" in body["capabilities"]


def test_api_show_unknown_model_404():
    r = client.post("/api/show", json={"name": "not-a-real-model"})
    assert r.status_code == 404


def test_api_version():
    r = client.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_v1_props():
    r = client.get("/v1/props")
    assert r.status_code == 200
    body = r.json()
    for key in ("default_generation_settings", "total_slots", "model_path", "chat_template"):
        assert key in body


def test_props():
    r = client.get("/props")
    assert r.status_code == 200
    body = r.json()
    for key in ("default_generation_settings", "total_slots", "model_path", "chat_template"):
        assert key in body


def test_api_v1_models_matches_v1_models():
    a = client.get("/api/v1/models")
    b = client.get("/v1/models")
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.json() == b.json()
