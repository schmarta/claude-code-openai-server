"""Compatibility shims for Ollama / llama.cpp-aware clients.

Frontends like Open WebUI probe a handful of non-OpenAI endpoints to detect
server capabilities (``/api/tags``, ``/api/show``, ``/v1/props``, ``/version`` …).
Without them the client falls back to a degraded mode or refuses the server.

These handlers reuse the OpenAI model list from :mod:`app.routes.models` so the
advertised models stay in one place; nothing here is hard-coded per model.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from app.openai_models import ModelList
from app.routes.models import list_models

router = APIRouter()


def _app_version() -> str:
    """Server version from installed package metadata, with a sane fallback."""
    for dist in ("claude-code-interface", "claude_code_interface"):
        try:
            return _pkg_version(dist)
        except PackageNotFoundError:
            continue
    return "0.1.0"


async def _model_ids() -> list[str]:
    """The advertised model ids, sourced from the OpenAI ``/v1/models`` handler."""
    models: ModelList = await list_models()
    return [card.id for card in models.data]


# Static "details" block reused by /api/tags and /api/show. Values are nominal —
# clients only need the shape to be present and well-formed.
_DETAILS = {
    "parent_model": "",
    "format": "gguf",
    "family": "claude",
    "families": ["claude"],
    "parameter_size": "",
    "quantization_level": "",
}


# ── Ollama: list models ───────────────────────────────────────────────────────


@router.get("/api/tags")
async def api_tags() -> dict:
    ids = await _model_ids()
    return {
        "models": [
            {
                "name": mid,
                "model": mid,
                "modified_at": "1970-01-01T00:00:00Z",
                "size": 0,
                "digest": "",
                "details": dict(_DETAILS),
            }
            for mid in ids
        ]
    }


# ── Ollama: show model ────────────────────────────────────────────────────────


class ShowRequest(BaseModel):
    model_config = {"extra": "ignore"}
    name: str | None = None
    model: str | None = None


@router.post("/api/show")
async def api_show(body: ShowRequest, response: Response) -> dict:
    requested = body.model or body.name
    ids = await _model_ids()
    if not requested or requested not in ids:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"error": f"model {requested!r} not found"}
    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": dict(_DETAILS),
        "model_info": {
            "general.architecture": "claude",
            "general.basename": requested,
        },
        "capabilities": ["completion"],
    }


# ── Ollama / llama.cpp: version ──────────────────────────────────────────────


@router.get("/api/version")
async def api_version() -> dict:
    return {"version": _app_version()}


@router.get("/version")
async def version() -> dict:
    return {"version": _app_version()}


# ── llama.cpp: server props ──────────────────────────────────────────────────


async def _props(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "default_generation_settings": {
            "n_ctx": 0,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "total_slots": 1,
        "model_path": settings.default_model,
        "chat_template": "",
    }


@router.get("/v1/props")
async def v1_props(request: Request) -> dict:
    return await _props(request)


@router.get("/props")
async def props(request: Request) -> dict:
    return await _props(request)


# ── Ollama-style alias for the OpenAI model list ─────────────────────────────


@router.get("/api/v1/models", response_model=ModelList)
async def api_v1_models() -> ModelList:
    return await list_models()
