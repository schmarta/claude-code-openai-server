"""Health/info endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "claude-code-interface", "version": __version__}


@router.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "claude-code-interface",
        "version": __version__,
        "openai_base": "/v1",
    }
